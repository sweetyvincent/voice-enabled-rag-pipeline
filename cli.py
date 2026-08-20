"""
Voice-Enabled RAG CLI Tool.

Usage:
  python cli.py --build-index           Build/rebuild FAISS index from MSMARCO-XI
  python cli.py --text "your question"  Text query mode
  python cli.py --voice                 Record from microphone (5 seconds)
  python cli.py --benchmark             Run latency benchmark
  python cli.py --stats                 Show latency statistics
"""
import argparse
import asyncio
import io
import sys
import logging
import os
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    import soundfile as sf
    HAVE_AUDIO = True
except ImportError:
    HAVE_AUDIO = False

# ANSI Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_result(result, query_type="Text"):
    """Pretty-print a PipelineResult to the terminal."""
    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}  {query_type} Query Results{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")

    print(f"\n{BOLD}Query:{RESET} {result.query}")

    if result.transcript and query_type == "Voice":
        print(f"{BOLD}Transcript:{RESET} {result.transcript}")

    if result.error:
        print(f"\n{RED}Error: {result.error}{RESET}")
        return

    # Retrieved chunks
    print(f"\n{BOLD}Retrieved Chunks ({len(result.retrieved_chunks)}):{RESET}")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        score_color = GREEN if chunk.score > 0.5 else YELLOW if chunk.score > 0.3 else RED
        print(f"  {DIM}[{i}]{RESET} {score_color}Score: {chunk.score:.4f}{RESET} "
              f"| Strategy: {CYAN}{chunk.strategy}{RESET}")
        display_text = chunk.text[:120].replace('\n', ' ')
        if len(chunk.text) > 120:
            display_text += "..."
        print(f"      {DIM}{display_text}{RESET}")

    # Answer
    print(f"\n{BOLD}Answer:{RESET}")
    if result.answer:
        print(f"  {result.answer}")
    else:
        print(f"  {DIM}(no answer generated){RESET}")

    # Guardrail verdict
    print(f"\n{BOLD}Guardrail Verdict:{RESET}")
    verdict = result.guardrail_verdict
    if verdict == "pass":
        print(f"  {GREEN}[PASS]{RESET}")
    elif "refuse" in verdict:
        print(f"  {RED}[REFUSE: {verdict.upper()}]{RESET}")
    elif "warn" in verdict:
        print(f"  {YELLOW}[WARN: {verdict.upper()}]{RESET}")
    else:
        print(f"  {verdict}")
    if result.guardrail_reason:
        print(f"  {DIM}Reason: {result.guardrail_reason}{RESET}")

    # Stage timing breakdown
    print(f"\n{BOLD}Stage Timing:{RESET}")
    for stage in result.stages:
        status_icon = (
            f"{GREEN}[PASS]{RESET}" if stage.status.value == "pass"
            else f"{RED}[FAIL]{RESET}" if stage.status.value == "fail"
            else f"{DIM}[SKIP]{RESET}"
        )
        print(f"  {status_icon} {stage.name:<20} {CYAN}{stage.latency_ms:>8.1f}ms{RESET}")
        if stage.error:
            print(f"    {RED}Error: {stage.error}{RESET}")

    # Total latency
    print(f"\n  {BOLD}{'-' * 35}{RESET}")
    if result.stt_latency_ms > 0:
        print(f"  STT Latency:       {CYAN}{result.stt_latency_ms:>8.1f}ms{RESET}")
    print(f"  Retrieval Latency: {CYAN}{result.retrieval_latency_ms:>8.1f}ms{RESET}")
    print(f"  Generation Latency:{CYAN}{result.generation_latency_ms:>8.1f}ms{RESET}")
    print(f"  {BOLD}Total Latency:   {CYAN}{result.total_latency_ms:>8.1f}ms{RESET}")

    # Under-200ms check (excluding STT)
    post_stt = result.total_latency_ms - result.stt_latency_ms
    target = 200.0
    if post_stt <= target:
        print(f"\n  {GREEN}[PASS] Post-STT latency ({post_stt:.1f}ms) is under {target:.0f}ms target{RESET}")
    else:
        print(f"\n  {YELLOW}[WARN] Post-STT latency ({post_stt:.1f}ms) exceeds {target:.0f}ms target{RESET}")

    print()


async def run_text_query(text: str):
    """Run a text query through the pipeline."""
    from harness.pipeline import RAGPipelineHarness
    from harness.models import PipelineRequest

    try:
        pipeline = RAGPipelineHarness()
    except Exception as e:
        print(f"{RED}Failed to initialize pipeline: {e}{RESET}")
        print(f"{YELLOW}Hint: Did you build the index? Run: python cli.py --build-index{RESET}")
        return

    request = PipelineRequest(text=text)
    result = await pipeline.execute(request)
    print_result(result, query_type="Text")


async def run_voice_query():
    """Record audio from microphone and run through pipeline."""
    if not HAVE_AUDIO:
        print(f"{RED}Audio recording not available.{RESET}")
        print(f"{YELLOW}Install required packages: pip install sounddevice soundfile{RESET}")
        return

    from harness.pipeline import RAGPipelineHarness
    from harness.models import PipelineRequest

    try:
        pipeline = RAGPipelineHarness()
    except Exception as e:
        print(f"{RED}Failed to initialize pipeline: {e}{RESET}")
        print(f"{YELLOW}Hint: Did you build the index? Run: python cli.py --build-index{RESET}")
        return

    duration = 5  # seconds
    fs = 16000    # Sample rate

    print(f"\n{YELLOW}🎤 Recording for {duration} seconds... Speak now!{RESET}")
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
    except Exception as e:
        print(f"{RED}Error recording audio: {e}{RESET}")
        return

    print(f"{GREEN}Recording finished. Processing...{RESET}")

    wav_io = io.BytesIO()
    sf.write(wav_io, recording, fs, format='WAV')
    wav_bytes = wav_io.getvalue()

    request = PipelineRequest(audio_bytes=wav_bytes)
    result = await pipeline.execute(request)
    print_result(result, query_type="Voice")


async def run_benchmark():
    """Run the latency benchmark."""
    from harness.pipeline import RAGPipelineHarness
    from analytics.benchmark import BenchmarkRunner

    try:
        pipeline = RAGPipelineHarness()
    except Exception as e:
        print(f"{RED}Failed to initialize pipeline: {e}{RESET}")
        print(f"{YELLOW}Hint: Did you build the index? Run: python cli.py --build-index{RESET}")
        return

    runner = BenchmarkRunner(pipeline)
    results = await runner.run(num_queries=50)

    print(f"\n{BOLD}Benchmark Results:{RESET}")
    print(json.dumps(results, indent=2, default=str))


def show_stats():
    """Show latency statistics from saved data."""
    from analytics.latency_tracker import LatencyTracker
    from config import get_settings

    settings = get_settings()
    tracker = LatencyTracker()

    try:
        tracker.load_from_file(settings.BENCHMARK_RESULTS_PATH)
        summary = tracker.get_summary()
        print(f"\n{BOLD}Latency Statistics:{RESET}")
        print(json.dumps(summary, indent=2, default=str))
    except FileNotFoundError:
        print(f"{YELLOW}No benchmark data found. Run: python cli.py --benchmark{RESET}")
    except Exception as e:
        print(f"{RED}Error loading stats: {e}{RESET}")


async def main():
    parser = argparse.ArgumentParser(
        description="Voice-Enabled RAG Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --build-index           Build the FAISS index
  python cli.py --text "What is TCP?"   Query by text
  python cli.py --voice                 Query by voice (5s recording)
  python cli.py --benchmark             Run latency benchmark (50 queries)
  python cli.py --stats                 Show latency statistics
        """
    )
    parser.add_argument(
        "--build-index", action="store_true",
        help="Build/rebuild FAISS index from MSMARCO-XI"
    )
    parser.add_argument("--text", type=str, help="Text query mode")
    parser.add_argument(
        "--voice", action="store_true",
        help="Record from microphone (5 seconds)"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run latency benchmark"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show latency statistics"
    )

    args = parser.parse_args()

    # Check API keys for commands that need them
    if args.text or args.voice or args.benchmark:
        try:
            from config import get_settings
            settings = get_settings()
        except Exception as e:
            print(f"{RED}Configuration error: {e}{RESET}")
            print(f"{YELLOW}Set SARVAM_API_KEY and GROQ_API_KEY environment variables or create a .env file{RESET}")
            return

    if args.build_index:
        print(f"{BOLD}Building FAISS index from MSMARCO-XI...{RESET}")
        try:
            from indexing.build_index import build_index
            build_index()
            print(f"\n{GREEN}[SUCCESS] Index built successfully!{RESET}")
        except Exception as e:
            print(f"{RED}Failed to build index: {e}{RESET}")
            import traceback
            traceback.print_exc()

    elif args.text:
        await run_text_query(args.text)

    elif args.voice:
        await run_voice_query()

    elif args.benchmark:
        await run_benchmark()

    elif args.stats:
        show_stats()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
