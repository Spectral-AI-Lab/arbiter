"""CLI entrypoint for starting the ARBITER inference server.

Usage:
    python -m arbiter.serve --port 8000 --model checkpoints/arbiter-272m
"""

import argparse
import logging
import os
import uvicorn
import torch

from ..modeling_arbiter import ArbiterModel
from .app import create_app

logger = logging.getLogger("arbiter.serve")


def main():
    default_model = os.environ.get("ARBITER_MODEL_PATH", "spectral-ai-lab/arbiter-272m")
    default_device = os.environ.get("ARBITER_DEVICE", "auto")

    parser = argparse.ArgumentParser(description="Launch the ARBITER inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--model", type=str, default=default_model, help=f"Local checkpoint or Hub ID (default: {default_model})")
    parser.add_argument("--device", type=str, default=default_device, help=f"Execution device (cuda:0, cpu, or auto, default: {default_device})")
    parser.add_argument("--batch-window", type=float, default=8.0, help="Micro-batching collection window in ms")
    parser.add_argument("--max-batch-size", type=int, default=16, help="Maximum batch size for dynamic batching")
    parser.add_argument("--max-batch-tokens", type=int, default=8192, help="Max tokens per forward batch")
    parser.add_argument("--max-attention-elements", type=int, default=16777216, help="Max attention elements per forward batch (default: 16777216)")
    parser.add_argument("--max-queue-size", type=int, default=1024, help="Maximum pending queue depth")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Initializing ARBITER from %s on %s...", args.model, device)
    model = ArbiterModel.from_pretrained(args.model, device=device)
    model.eval()

    app = create_app(
        model=model,
        model_id=args.model,
        batch_window_ms=args.batch_window,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_attention_elements=args.max_attention_elements,
        max_queue_size=args.max_queue_size,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
