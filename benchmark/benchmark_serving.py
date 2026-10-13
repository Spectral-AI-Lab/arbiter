"""Benchmark script for ARBITER online serving throughput and latency.

Evaluates:
- Concurrency scaling (1, 4, 8, 16, 32 concurrent clients)
- Batch window impact (0ms immediate dispatch vs 8ms micro-batching)
- End-to-end p50/p95 latencies, queue saturation, and decisions/second
- Peak memory footprint on CPU and GPU
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import torch

from arbiter.modeling_arbiter import ArbiterModel
from arbiter.serve.app import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark.serving")

# Realistic workload payloads
WORKLOAD = [
    {
        "endpoint": "/v1/systemone",
        "payload": {
            "state": {"ticket": "My transaction was declined three times at supermarket terminal."},
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Classify banking issue",
                    "criteria": {
                        "card_declined": "Declined card or POS error",
                        "atm_error": "ATM cash withdrawal failure",
                        "transfer_delay": "Wire transfer pending",
                    },
                },
                "fraud_alert": {
                    "type": "noul",
                    "instructions": "Is this potentially an account security block?",
                    "criteria": {"true": "Card blocked due to suspected unauthorized activity"},
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Rate customer urgency level",
                    "criteria": ["routine", "moderate", "critical"],
                },
            },
        },
        "decision_count": 3,
    },
    {
        "endpoint": "/decide",
        "payload": {
            "context": "Customer wants to dispute an international charge from a merchant in London.",
            "instruction": "Route query to resolution desk",
            "criteria": {
                "dispute": "Charge dispute and chargebacks",
                "exchange_rate": "Foreign transaction fees and FX markup",
                "card_lost": "Lost or stolen card replacement",
            },
            "question_type": "choice",
        },
        "decision_count": 1,
    },
    {
        "endpoint": "/v1/systemone",
        "payload": {
            "state": "The parcel tracking indicates delivered to front porch, but nothing is there.",
            "questions": {
                "issue": {
                    "type": "choice",
                    "instructions": "Identify shipment problem",
                    "criteria": {
                        "lost_package": "Missing package marked delivered",
                        "damaged_goods": "Damaged contents on arrival",
                        "delay": "Carrier transit delay",
                    },
                },
                "requires_refund": {
                    "type": "noul",
                    "instructions": "Is customer requesting claim investigation?",
                },
            },
        },
        "decision_count": 2,
    },
    {
        "endpoint": "/decide",
        "payload": {
            "context": "My account password was reset without my knowledge, cannot login anymore.",
            "instruction": "Is account compromise imminent?",
            "question_type": "noul",
        },
        "decision_count": 1,
    },
]


async def run_client_worker(
    client: httpx.AsyncClient,
    queue: asyncio.Queue,
    latencies: list[float],
    status_codes: list[int],
    workload_executed: list[dict[str, Any]],
) -> None:
    while not queue.empty():
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        endpoint = item["endpoint"]
        payload = item["payload"]

        start = time.perf_counter()
        try:
            resp = await client.post(endpoint, json=payload, timeout=30.0)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            status_codes.append(resp.status_code)
            workload_executed.append(item)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            status_codes.append(500)
            workload_executed.append(item)
        finally:
            queue.task_done()


async def run_benchmark_trial(
    app,
    concurrency: int,
    num_requests: int,
    device: str = "cpu",
) -> dict[str, Any]:
    # Reset metrics and per-trial GPU memory counters
    app.state.metrics._latencies_ms.clear()
    app.state.metrics._batch_sizes.clear()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.reset_peak_memory_stats()

    # Pre-fill request queue
    queue: asyncio.Queue = asyncio.Queue()
    for i in range(num_requests):
        item = WORKLOAD[i % len(WORKLOAD)]
        queue.put_nowait(item)

    latencies: list[float] = []
    status_codes: list[int] = []
    workload_executed: list[dict[str, Any]] = []

    # Run trial using httpx AsyncClient with ASGITransport (in-process engine evaluation)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        start_wall = time.perf_counter()
        workers = [
            asyncio.create_task(run_client_worker(client, queue, latencies, status_codes, workload_executed))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)
        wall_time_s = time.perf_counter() - start_wall

    # Track successes and successful decisions
    success_requests = sum(1 for c in status_codes if c == 200)
    error_count = len(status_codes) - success_requests
    successful_decisions = sum(
        item["decision_count"]
        for item, code in zip(workload_executed, status_codes)
        if code == 200
    )
    total_attempted_decisions = sum(item["decision_count"] for item in workload_executed)

    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    p50 = lat_sorted[int(n * 0.50)] if n else 0.0
    p90 = lat_sorted[min(int(n * 0.90), n - 1)] if n else 0.0
    p95 = lat_sorted[min(int(n * 0.95), n - 1)] if n else 0.0

    req_per_sec = success_requests / wall_time_s if wall_time_s > 0 else 0.0
    dec_per_sec = successful_decisions / wall_time_s if wall_time_s > 0 else 0.0

    # Resource metrics
    import resource
    cpu_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    peak_gpu_mib = 0.0
    if torch.cuda.is_available() and "cuda" in device:
        peak_gpu_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "success_requests": success_requests,
        "errors": error_count,
        "successful_decisions": successful_decisions,
        "total_attempted_decisions": total_attempted_decisions,
        "wall_time_s": round(wall_time_s, 3),
        "requests_per_sec": round(req_per_sec, 1),
        "decisions_per_sec": round(dec_per_sec, 1),
        "latency_p50_ms": round(p50, 2),
        "latency_p90_ms": round(p90, 2),
        "latency_p95_ms": round(p95, 2),
        "avg_batch_size": app.state.metrics.snapshot().get("avg_batch_size", 1.0),
        "cpu_rss_mib": round(cpu_rss_mib, 1),
        "gpu_peak_mib": round(peak_gpu_mib, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="ARBITER Serving Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/arbiter-272m")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--concurrencies", type=str, default="1,4,8,16,32")
    parser.add_argument("--batch-windows", type=str, default="0,8")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    concurrency_list = [int(c.strip()) for c in args.concurrencies.split(",")]
    batch_windows = [float(w.strip()) for w in args.batch_windows.split(",")]

    logger.info("Loading model from %s on %s...", args.checkpoint, args.device)
    model = ArbiterModel.from_pretrained(args.checkpoint, device=args.device)
    model.eval()

    if torch.cuda.is_available() and "cuda" in args.device:
        torch.cuda.reset_peak_memory_stats()

    all_results = []

    print("\n" + "=" * 90)
    print(f"ARBITER ONLINE SERVING BENCHMARK — Device: {args.device}")
    print(f"Requests per run: {args.num_requests} mixed (/v1/systemone + /decide)")
    print("=" * 90)

    for window_ms in batch_windows:
        mode_str = f"Window {window_ms:.0f}ms (Micro-Batching)" if window_ms > 0 else "Window 0ms (Immediate Dispatch)"
        print(f"\n--- Mode: {mode_str} ---")

        # Create app with this batch window
        app = create_app(
            model=model,
            model_id="spectral-ai-lab/arbiter-272m",
            batch_window_ms=window_ms,
            max_batch_size=16,
        )

        async def run_suite():
            suite_results = []
            # Start lifespan
            async with app.router.lifespan_context(app):
                # Warmup
                await run_benchmark_trial(app, concurrency=1, num_requests=4, device=args.device)
                for conc in concurrency_list:
                    res = await run_benchmark_trial(app, concurrency=conc, num_requests=args.num_requests, device=args.device)
                    res["batch_window_ms"] = window_ms
                    suite_results.append(res)
                    print(
                        f"Conc: {conc:2d} | "
                        f"Throughput: {res['decisions_per_sec']:6.1f} dec/s ({res['requests_per_sec']:5.1f} req/s) | "
                        f"Lat p50: {res['latency_p50_ms']:5.1f} ms | "
                        f"Lat p95: {res['latency_p95_ms']:5.1f} ms | "
                        f"Avg Batch: {res['avg_batch_size']:4.1f} | "
                        f"RSS: {res['cpu_rss_mib']:5.1f} MB | "
                        f"GPU: {res['gpu_peak_mib']:5.1f} MB | "
                        f"Errors: {res['errors']}"
                    )
            return suite_results

        results = asyncio.run(run_suite())
        all_results.extend(results)

    peak_vram_mib = 0.0
    if torch.cuda.is_available() and "cuda" in args.device:
        peak_vram_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"\nPeak GPU Memory Allocated: {peak_vram_mib:.1f} MiB")

    # Generate markdown table summary
    print("\n" + "=" * 90)
    print("EMPIRICAL SERVING SUMMARY TABLE")
    print("=" * 90)
    print("| Batch Window | Concurrency | Throughput (dec/s) | Throughput (req/s) | Latency p50 | Latency p95 | Avg Batch |")
    print("|---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in all_results:
        w_label = f"{r['batch_window_ms']:.0f} ms"
        print(f"| {w_label:12s} | {r['concurrency']:11d} | {r['decisions_per_sec']:18.1f} | {r['requests_per_sec']:18.1f} | {r['latency_p50_ms']:9.1f} ms | {r['latency_p95_ms']:9.1f} ms | {r['avg_batch_size']:9.1f} |")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "results": all_results,
                "peak_vram_mib": peak_vram_mib,
                "device": args.device,
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
