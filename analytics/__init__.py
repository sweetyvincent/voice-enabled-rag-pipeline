"""
Analytics module for tracking latency and running benchmarks on the RAG pipeline.
"""

from analytics.latency_tracker import LatencyTracker
from analytics.benchmark import BenchmarkRunner

__all__ = ["LatencyTracker", "BenchmarkRunner"]
