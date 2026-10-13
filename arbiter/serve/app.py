"""FastAPI application for ARBITER inference serving.

Implements:
- POST /v1/systemone: vLLM Structured Decisions specification
- POST /predict: Backward-compatible alias for Jev / Laya / systemone
- POST /decide: Lightweight native flat ARBITER decision endpoint
- GET /health: Liveness and model status
- GET /metrics: Real-time operational latencies and throughput
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..modeling_arbiter import ArbiterModel
from .adapter import (
    assemble_systemone_response,
    build_legacy_predict_answer,
    build_systemone_answer,
    prepare_question_task,
    serialize_state,
)
from .metrics import ServingMetrics
from .schemas import (
    DecideRequest,
    DecideResponse,
    HealthResponse,
    MetricsResponse,
    SystemOneRequest,
    SystemOneResponse,
)
from .scheduler import DynamicBatchScheduler, QueueFullError

logger = logging.getLogger("arbiter.serve")


def get_canonical_model_id(served_id: str) -> str:
    """Resolve the canonical public Hugging Face model identifier."""
    name = Path(served_id).name.lower()
    if "typed" in name or "typed" in served_id.lower():
        return "spectral-ai-lab/arbiter-272m-typed"
    return "spectral-ai-lab/arbiter-272m"


def validate_model_id(requested_model: Optional[str], served_id: str) -> str:
    """Validate requested model against the actually loaded model and its valid aliases."""
    canonical_id = get_canonical_model_id(served_id)
    if not requested_model:
        return canonical_id

    is_served_typed = "typed" in canonical_id

    if is_served_typed:
        valid_aliases = {
            served_id,
            Path(served_id).name,
            "spectral-ai-lab/arbiter-272m-typed",
            "arbiter-272m-typed",
        }
    else:
        valid_aliases = {
            served_id,
            Path(served_id).name,
            "spectral-ai-lab/arbiter-272m",
            "arbiter-272m",
            "arbiter",
        }

    if requested_model not in valid_aliases:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requested model '{requested_model}' does not match served model '{served_id}'.",
        )
    return canonical_id


def create_app(
    model: Optional[ArbiterModel] = None,
    model_id: str = "spectral-ai-lab/arbiter-272m",
    batch_window_ms: float = 8.0,
    max_batch_size: int = 16,
    max_batch_tokens: int = 8192,
    max_attention_elements: int = 16777216,
    max_queue_size: int = 1024,
) -> FastAPI:
    """Create and configure the FastAPI serving application."""
    metrics = ServingMetrics(max_history=2000)
    loaded_model: Optional[ArbiterModel] = model
    scheduler: Optional[DynamicBatchScheduler] = None

    if loaded_model is not None:
        scheduler = DynamicBatchScheduler(
            model=loaded_model,
            metrics=metrics,
            batch_window_ms=batch_window_ms,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            max_attention_elements=max_attention_elements,
            max_queue_size=max_queue_size,
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal scheduler, loaded_model
        # If model not provided, load from env or default
        if loaded_model is None:
            ckpt_path = os.environ.get("ARBITER_MODEL_PATH", model_id)
            device = os.environ.get("ARBITER_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
            logger.info("Loading ARBITER model from %s on %s...", ckpt_path, device)
            loaded_model = ArbiterModel.from_pretrained(ckpt_path, device=device)
            loaded_model.eval()

        if scheduler is None:
            scheduler = DynamicBatchScheduler(
                model=loaded_model,
                metrics=metrics,
                batch_window_ms=batch_window_ms,
                max_batch_size=max_batch_size,
                max_batch_tokens=max_batch_tokens,
                max_attention_elements=max_attention_elements,
                max_queue_size=max_queue_size,
            )
        await scheduler.start()
        app.state.scheduler = scheduler
        app.state.model = loaded_model
        app.state.metrics = metrics
        app.state.model_id = model_id
        yield
        await scheduler.stop()

    app = FastAPI(
        title="ARBITER Inference Service",
        description="High-performance serving for ARBITER Structured Decisions & System 1 Routing",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.scheduler = scheduler
    app.state.model = loaded_model
    app.state.metrics = metrics
    app.state.model_id = model_id

    # Convert validation errors to HTTP 400 per vLLM Structured Decisions specification
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        sanitized_errors = []
        for err in exc.errors():
            err_dict = dict(err)
            if "ctx" in err_dict and isinstance(err_dict["ctx"], dict):
                err_dict["ctx"] = {k: str(v) for k, v in err_dict["ctx"].items()}
            sanitized_errors.append(err_dict)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": sanitized_errors},
        )

    @app.exception_handler(QueueFullError)
    async def queue_full_exception_handler(request: Request, exc: QueueFullError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers={"Retry-After": "1"},
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        device_str = "unloaded"
        is_loaded = loaded_model is not None
        if is_loaded:
            try:
                device_str = str(next(loaded_model.parameters()).device)
            except Exception:
                device_str = "cpu"
        return HealthResponse(
            status="healthy",
            model_id=app.state.model_id if hasattr(app.state, "model_id") else model_id,
            device=device_str,
            loaded=is_loaded,
            version=__version__,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        return MetricsResponse(**metrics.snapshot())

    async def _execute_questions(
        req: SystemOneRequest,
    ) -> tuple[str, list[str], list[dict[str, Any]], dict[str, int]]:
        start_time = time.monotonic()
        active_scheduler: DynamicBatchScheduler = app.state.scheduler
        if active_scheduler._worker_task is None or active_scheduler._worker_task.done():
            await active_scheduler.start()

        served_id = app.state.model_id if hasattr(app.state, "model_id") else model_id
        canonical_id = validate_model_id(req.model, served_id)

        # Check queue capacity upfront for all questions atomically
        try:
            active_scheduler.check_capacity(len(req.questions))
        except QueueFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})

        state_str = serialize_state(req.state)

        # Prepare and enqueue a task for each question atomically
        task_futures = []
        qid_order = []
        try:
            for qid, question in req.questions.items():
                payload = prepare_question_task(state_str, qid, question)
                fut = active_scheduler.submit(payload)
                task_futures.append(fut)
                qid_order.append(qid)
        except (ValueError, TypeError) as exc:
            for f in task_futures:
                f.cancel()
            raise HTTPException(status_code=400, detail=str(exc))
        except QueueFullError as exc:
            for f in task_futures:
                f.cancel()
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})
        except Exception as exc:
            for f in task_futures:
                f.cancel()
            raise HTTPException(status_code=500, detail=f"Request admission failed: {exc}")

        # Await all decisions; cancel remaining tasks on any failure
        try:
            results = await asyncio.gather(*task_futures)
        except (ValueError, TypeError) as exc:
            for f in task_futures:
                if not f.done():
                    f.cancel()
            raise HTTPException(status_code=400, detail=str(exc))
        except QueueFullError as exc:
            for f in task_futures:
                if not f.done():
                    f.cancel()
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})
        except Exception as exc:
            for f in task_futures:
                if not f.done():
                    f.cancel()
            raise HTTPException(status_code=500, detail=f"Inference execution failed: {exc}")

        tokens_by_qid = {
            qid: raw_res.get("input_tokens", 0)
            for qid, raw_res in zip(qid_order, results)
        }
        duration_ms = (time.monotonic() - start_time) * 1000.0
        metrics.record_request(duration_ms)
        return canonical_id, qid_order, results, tokens_by_qid

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    async def system_one_endpoint(req: SystemOneRequest):
        """Standardized vLLM Structured Decisions endpoint.
        
        Evaluates typed decisions (choice, noul, score) over context in a single
        non-causal forward pass without autoregressive generation.
        """
        canonical_id, qid_order, results, tokens_by_qid = await _execute_questions(req)
        answers_by_qid = {}
        for qid, raw_res in zip(qid_order, results):
            question = req.questions[qid]
            answers_by_qid[qid] = build_systemone_answer(question, raw_res)
        return assemble_systemone_response(req, canonical_id, answers_by_qid, tokens_by_qid)

    @app.post("/predict")
    async def predict_endpoint(req: SystemOneRequest):
        """Backward-compatible endpoint for Jev / Laya / native model.predict contract.
        
        Preserves native boolean identifiers ('false' and 'true') and direct answer dictionaries.
        """
        _, qid_order, results, _ = await _execute_questions(req)
        answers_by_qid = {}
        for qid, raw_res in zip(qid_order, results):
            question = req.questions[qid]
            answers_by_qid[qid] = build_legacy_predict_answer(question, raw_res)
        return {"answers": answers_by_qid}

    @app.post("/decide", response_model=DecideResponse)
    async def decide_endpoint(req: DecideRequest):
        """Native ARBITER flat decision endpoint."""
        start_time = time.monotonic()
        active_scheduler: DynamicBatchScheduler = app.state.scheduler
        if active_scheduler._worker_task is None or active_scheduler._worker_task.done():
            await active_scheduler.start()

        try:
            active_scheduler.check_capacity(1)
        except QueueFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})

        exec_payload = {
            "context": req.context,
            "instruction": req.instruction,
            "criteria": req.criteria,
            "question_type": req.question_type,
            "max_length": req.max_length,
            "allow_truncation": req.allow_truncation,
        }

        try:
            fut = active_scheduler.submit(exec_payload)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except QueueFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Request submission failed: {exc}")

        try:
            res = await fut
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except QueueFullError as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"})
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference execution failed: {exc}")

        duration_ms = (time.monotonic() - start_time) * 1000.0
        metrics.record_request(duration_ms)

        return DecideResponse(
            selected_key=res["selected_key"],
            probabilities=res["probabilities"],
            confidence=res["confidence"],
            score=res.get("score"),
            noul=res.get("noul"),
            input_tokens=res.get("input_tokens", 0),
            truncated=res.get("truncated", False),
        )

    return app
