"""
Benchmark runner for measuring pipeline latency with P50/P70/P90/P99/P100 analytics.
"""
import asyncio
import logging
import random
import json
import time
from typing import List, Dict, Any, Optional

from analytics.latency_tracker import LatencyTracker
from harness.models import PipelineRequest
from config import get_settings

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

logger = logging.getLogger(__name__)

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Sample test queries covering MSMARCO domain topics
BENCHMARK_TEST_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does photosynthesis work?",
    "Who built the Taj Mahal?",
    "What did Chandrayaan-3 achieve?",
    "When did India gain independence?",
    "What are the symptoms of dengue fever?",
    "How is biryani cooked?",
    "What is general relativity?",
    "How do vaccines work?",
    "Why is sleep important?",
    "What is the Indus Valley Civilization?",
    "Where does the Ganges river originate?",
    "What are the Western Ghats?",
    "What is UPI?",
    "How do vector databases work?",
    "How does fermentation work in dosa batter?",
    "What is the Chola dynasty?",
    "What are the health benefits of exercise?"
]


class BenchmarkRunner:
    """Runs latency benchmarks across a set of test queries."""

    def __init__(self, pipeline):
        """Initialize with a RAGPipelineHarness instance."""
        self.pipeline = pipeline
        self.latency_tracker = LatencyTracker(window_size=10000)

    async def run(self, num_queries: int = 50) -> Dict[str, Any]:
        """Run the benchmark on queries from MSMARCO.

        Args:
            num_queries: Number of test queries to run.

        Returns:
            Summary dict with P50/P70/P90/P99/P100 latency numbers.
        """
        settings = get_settings()

        print(f"\n{BOLD}Loading benchmark queries...{RESET}")
        queries = list(BENCHMARK_TEST_QUERIES)

        # Duplicate and shuffle to reach target num_queries
        while len(queries) < num_queries:
            queries.extend(BENCHMARK_TEST_QUERIES)
        random.shuffle(queries)
        queries = queries[:num_queries]

        print(f"{GREEN}Loaded {len(queries)} test queries. Executing benchmark...{RESET}\n")

        # Run queries
        success_count = 0
        fail_count = 0
        start_time = time.time()

        for i, query in enumerate(queries, 1):
            try:
                req = PipelineRequest(text=query)
                result = await self.pipeline.execute(req)
                self.latency_tracker.record(result)

                status = (
                    f"{GREEN}[PASS]{RESET}" if result.success
                    else f"{RED}[FAIL]{RESET}"
                )
                latency = result.total_latency_ms
                print(
                    f"  {status} [{i:3d}/{len(queries)}] "
                    f"{CYAN}{latency:>7.1f}ms{RESET} "
                    f"{DIM}{query[:60]}...{RESET}"
                )

                if result.success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                logger.error(f"Query {i} failed: {e}")

        total_time = time.time() - start_time

        # Compute summary
        summary = self.latency_tracker.get_summary()

        summary["benchmark_metadata"] = {
            "num_queries": len(queries),
            "success_count": success_count,
            "fail_count": fail_count,
            "total_benchmark_time_s": round(total_time, 2),
            "avg_query_time_s": round(total_time / max(len(queries), 1), 4)
        }

        # Get fastest/slowest samples
        records = sorted(
            self.latency_tracker.records,
            key=lambda x: x.get("total_latency_ms", 0.0)
        )
        summary["fastest_3"] = [
            {"query": r.get("query", ""), "latency_ms": r.get("total_latency_ms", 0)}
            for r in records[:3]
        ]
        summary["slowest_3"] = [
            {"query": r.get("query", ""), "latency_ms": r.get("total_latency_ms", 0)}
            for r in records[-3:]
        ]

        # Save results
        try:
            self.latency_tracker.save_to_file(settings.BENCHMARK_RESULTS_PATH)
            print(f"\n{GREEN}Benchmark results saved to {settings.BENCHMARK_RESULTS_PATH}{RESET}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

        # Print histogram
        latencies = [
            r.get("total_latency_ms", 0.0)
            for r in self.latency_tracker.records
            if "total_latency_ms" in r
        ]
        self._print_histogram(latencies)

        # Print formatted summary table
        print(self._format_results(summary))

        return summary

    def _print_histogram(self, latencies: List[float], bins: int = 10) -> None:
        """Create ASCII art histogram showing latency distribution."""
        if not latencies:
            print("No latency data to display.")
            return

        min_val = min(latencies)
        max_val = max(latencies)

        if min_val == max_val:
            print(f"All latencies are {min_val:.2f} ms")
            return

        step = (max_val - min_val) / max(bins, 1)
        counts = [0] * bins

        for lat in latencies:
            idx = min(int((lat - min_val) / step), bins - 1)
            counts[idx] += 1

        max_count = max(counts) if counts else 1

        print(f"\n{BOLD}Latency Distribution Histogram (ms):{RESET}")
        print(f"{'-' * 55}")
        for i in range(bins):
            bin_start = min_val + i * step
            bin_end = bin_start + step
            bar_len = int((counts[i] / max_count) * 30)
            bar = '|' * bar_len
            print(f"  {bin_start:6.1f}-{bin_end:6.1f}ms | {counts[i]:3d} {CYAN}{bar}{RESET}")
        print(f"{'-' * 55}")

    def _format_results(self, summary: Dict[str, Any]) -> str:
        """Format results as a clean ASCII table."""
        lines = []
        lines.append(f"\n{BOLD}{'=' * 55}{RESET}")
        lines.append(f"{BOLD}  LATENCY ANALYTICS & BENCHMARK SUMMARY{RESET}")
        lines.append(f"{BOLD}{'=' * 55}{RESET}")

        meta = summary.get("benchmark_metadata", {})
        lines.append(f"  Total Queries: {meta.get('num_queries', 0)}")
        lines.append(f"  Success Rate:  {GREEN}{summary.get('success_rate', 1.0)*100:.1f}%{RESET}")
        lines.append(f"  Total Time:    {meta.get('total_benchmark_time_s', 0):.2f}s")
        lines.append(f"  Avg Query:     {meta.get('avg_query_time_s', 0)*1000:.2f}ms")

        # Overall latency percentiles
        percentiles = summary.get("percentiles", {})
        total = percentiles.get("total_latency_ms", {})

        lines.append(f"\n{BOLD}  Overall Post-STT Latency Metrics (ms):{RESET}")
        lines.append(f"  {'-' * 40}")
        for pct in ["p50", "p70", "p90", "p99", "p100"]:
            val = total.get(pct, 0.0)
            color = GREEN if val < 200 else YELLOW if val < 500 else RED
            lines.append(f"    {pct.upper():>5s}:  {color}{val:>8.2f} ms{RESET}")
        lines.append(f"    MEAN: {CYAN}{total.get('mean', 0.0):>8.2f} ms{RESET}")

        # Per-stage breakdown
        stage_pcts = summary.get("stage_percentiles", {})
        if stage_pcts:
            lines.append(f"\n{BOLD}  Per-Stage Latency Breakdown (P50 / P70 / P100):{RESET}")
            lines.append(f"  {'-' * 40}")
            for stage_name, pcts in stage_pcts.items():
                p50 = pcts.get("p50", 0.0)
                p70 = pcts.get("p70", 0.0)
                p100 = pcts.get("p100", 0.0)
                lines.append(
                    f"    {stage_name:20s}: "
                    f"{CYAN}{p50:6.2f}ms{RESET} / "
                    f"{CYAN}{p70:6.2f}ms{RESET} / "
                    f"{CYAN}{p100:6.2f}ms{RESET}"
                )

        lines.append(f"\n{BOLD}{'=' * 55}{RESET}")
        return "\n".join(lines)


async def main():
    """Standalone benchmark entry point."""
    logging.basicConfig(level=logging.INFO)

    try:
        from harness.pipeline import RAGPipelineHarness
    except ImportError:
        logger.error("Could not import RAGPipelineHarness.")
        return

    logger.info("Initializing pipeline...")
    pipeline = RAGPipelineHarness()

    runner = BenchmarkRunner(pipeline=pipeline)

    try:
        await runner.run(num_queries=50)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
