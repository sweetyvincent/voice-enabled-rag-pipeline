"""
Latency analytics tracker — records pipeline execution timings and computes percentiles.
"""
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class LatencyTracker:
    """Tracks pipeline latency with rolling window and percentile computation."""

    def __init__(self, window_size: int = 100):
        """Initialize with a rolling window size for records."""
        self.window_size = window_size
        self.records: List[Dict[str, Any]] = []

    def record(self, pipeline_result) -> None:
        """Extract and store timing data from a PipelineResult.

        Args:
            pipeline_result: A PipelineResult object from the harness.
        """
        try:
            # Extract per-stage timings from the stages list
            stage_timings = {}
            stages = getattr(pipeline_result, "stages", [])
            for stage in stages:
                name = getattr(stage, "name", "unknown")
                latency = getattr(stage, "latency_ms", 0.0)
                stage_timings[name] = latency

            record_data = {
                "timestamp": datetime.now().isoformat(),
                "query": getattr(pipeline_result, "query", ""),
                "success": getattr(pipeline_result, "success", False),
                "total_latency_ms": getattr(pipeline_result, "total_latency_ms", 0.0),
                "stt_latency_ms": getattr(pipeline_result, "stt_latency_ms", 0.0),
                "retrieval_latency_ms": getattr(pipeline_result, "retrieval_latency_ms", 0.0),
                "generation_latency_ms": getattr(pipeline_result, "generation_latency_ms", 0.0),
                "guardrail_verdict": getattr(pipeline_result, "guardrail_verdict", ""),
                "num_chunks_retrieved": len(getattr(pipeline_result, "retrieved_chunks", [])),
                "stages": stage_timings,
            }

            self.records.append(record_data)

            # Keep within rolling window size
            if len(self.records) > self.window_size:
                self.records.pop(0)

        except Exception as e:
            logger.error(f"Failed to record latency: {e}")

    def get_percentiles(self, metric: str = 'total_latency_ms') -> Dict[str, float]:
        """Compute P50/P70/P90/P99/P100 for a specific metric."""
        if not self.records:
            return {
                "p50": 0.0, "p70": 0.0, "p90": 0.0,
                "p99": 0.0, "p100": 0.0, "count": 0, "mean": 0.0
            }

        values = [r.get(metric, 0.0) for r in self.records if metric in r]
        if not values:
            return {
                "p50": 0.0, "p70": 0.0, "p90": 0.0,
                "p99": 0.0, "p100": 0.0, "count": 0, "mean": 0.0
            }

        arr = np.array(values)
        return {
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p70": round(float(np.percentile(arr, 70)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "p100": round(float(np.percentile(arr, 100)), 2),
            "count": len(values),
            "mean": round(float(np.mean(arr)), 2),
        }

    def get_stage_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Compute percentiles broken down by pipeline stage."""
        if not self.records:
            return {}

        stages_data: Dict[str, List[float]] = {}
        for record in self.records:
            stages = record.get("stages", {})
            for stage_name, ms in stages.items():
                if stage_name not in stages_data:
                    stages_data[stage_name] = []
                stages_data[stage_name].append(ms)

        result = {}
        for stage_name, values in stages_data.items():
            arr = np.array(values)
            result[stage_name] = {
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p70": round(float(np.percentile(arr, 70)), 2),
                "p90": round(float(np.percentile(arr, 90)), 2),
                "p99": round(float(np.percentile(arr, 99)), 2),
                "p100": round(float(np.percentile(arr, 100)), 2),
                "count": len(values),
                "mean": round(float(np.mean(arr)), 2),
            }
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Comprehensive summary with percentiles for all tracked metrics."""
        if not self.records:
            return {"total_queries": 0, "success_rate": 0.0}

        total = len(self.records)
        successes = sum(1 for r in self.records if r.get("success", False))

        return {
            "total_queries": total,
            "success_rate": round(successes / total, 4) if total > 0 else 0.0,
            "percentiles": {
                "total_latency_ms": self.get_percentiles("total_latency_ms"),
                "stt_latency_ms": self.get_percentiles("stt_latency_ms"),
                "retrieval_latency_ms": self.get_percentiles("retrieval_latency_ms"),
                "generation_latency_ms": self.get_percentiles("generation_latency_ms"),
            },
            "stage_percentiles": self.get_stage_percentiles(),
        }

    def save_to_file(self, path: str) -> None:
        """Save all raw records and computed summary to JSON."""
        try:
            data = {
                "records": self.records,
                "summary": self.get_summary(),
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.records)} latency records to {path}")
        except Exception as e:
            logger.error(f"Failed to save latency records: {e}")

    def load_from_file(self, path: str) -> None:
        """Load records from a previously saved JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.records = data.get("records", [])
        logger.info(f"Loaded {len(self.records)} records from {path}")
