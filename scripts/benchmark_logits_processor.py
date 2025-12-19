#!/usr/bin/env python3
"""
Benchmark the DiplomacyLogitsProcessor overhead.

This script measures the time spent in the logits processor's apply() method
to determine if it's a significant bottleneck in inference.

Key measurements:
- Time per apply() call
- Time in _update_request_state()
- Time in _restrict_logits_to_ids()
- Trie build time

Comparison baseline:
- vLLM decode step: ~2-5ms per token on A100
- If logits processor adds <5% overhead, it's acceptable
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MockOutputTokenIds:
    """Mock vLLM output token IDs."""

    token_ids: list[int]

    def __iter__(self):
        return iter(self.token_ids)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        return self.token_ids[idx]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    num_calls: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float

    def print_report(self, label: str):
        print(f"\n{label}:")
        print(f"  Calls: {self.num_calls}")
        print(f"  Total: {self.total_time_ms:.2f}ms")
        print(f"  Avg: {self.avg_time_ms:.3f}ms")
        print(f"  Min: {self.min_time_ms:.3f}ms")
        print(f"  Max: {self.max_time_ms:.3f}ms")


def get_realistic_valid_moves() -> dict[str, list[str]]:
    """
    Get realistic valid moves for a Diplomacy position.
    This represents a typical mid-game position with ~15 units.
    """
    return {
        "A PAR": [
            "A PAR H",
            "A PAR - BUR",
            "A PAR - PIC",
            "A PAR - GAS",
            "A PAR - BRE",
            "A PAR S A MAR - BUR",
            "A PAR S F BRE - MAO",
        ],
        "A MAR": [
            "A MAR H",
            "A MAR - BUR",
            "A MAR - PIE",
            "A MAR - SPA",
            "A MAR - GAS",
            "A MAR S A PAR - BUR",
        ],
        "F BRE": [
            "F BRE H",
            "F BRE - MAO",
            "F BRE - ENG",
            "F BRE - PIC",
            "F BRE S A PAR - PIC",
        ],
        "A MUN": [
            "A MUN H",
            "A MUN - BUR",
            "A MUN - RUH",
            "A MUN - KIE",
            "A MUN - BOH",
            "A MUN - SIL",
            "A MUN - TYR",
        ],
        "A BER": [
            "A BER H",
            "A BER - KIE",
            "A BER - PRU",
            "A BER - SIL",
            "A BER - MUN",
        ],
        "F KIE": [
            "F KIE H",
            "F KIE - HEL",
            "F KIE - HOL",
            "F KIE - DEN",
            "F KIE - BAL",
        ],
        "A VIE": [
            "A VIE H",
            "A VIE - BOH",
            "A VIE - GAL",
            "A VIE - BUD",
            "A VIE - TRI",
            "A VIE - TYR",
        ],
        "A BUD": [
            "A BUD H",
            "A BUD - VIE",
            "A BUD - GAL",
            "A BUD - RUM",
            "A BUD - SER",
            "A BUD - TRI",
        ],
        "F TRI": [
            "F TRI H",
            "F TRI - VEN",
            "F TRI - ADR",
            "F TRI - ALB",
        ],
        "A ROM": [
            "A ROM H",
            "A ROM - VEN",
            "A ROM - TUS",
            "A ROM - NAP",
            "A ROM - APU",
        ],
        "F NAP": [
            "F NAP H",
            "F NAP - TYS",
            "F NAP - ION",
            "F NAP - APU",
            "F NAP - ROM",
        ],
        "A WAR": [
            "A WAR H",
            "A WAR - PRU",
            "A WAR - SIL",
            "A WAR - GAL",
            "A WAR - UKR",
            "A WAR - MOS",
            "A WAR - LVN",
        ],
        "A MOS": [
            "A MOS H",
            "A MOS - SEV",
            "A MOS - UKR",
            "A MOS - WAR",
            "A MOS - LVN",
            "A MOS - STP",
        ],
        "F SEV": [
            "F SEV H",
            "F SEV - BLA",
            "F SEV - RUM",
            "F SEV - ARM",
        ],
        "A CON": [
            "A CON H",
            "A CON - BUL",
            "A CON - SMY",
            "A CON - ANK",
        ],
    }


def benchmark_trie_build(
    processor: Any, valid_moves: dict, num_iterations: int = 100
) -> BenchmarkResult:
    """Benchmark trie building time."""
    times = []

    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = processor._build_trie(valid_moves)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        num_calls=num_iterations,
        total_time_ms=sum(times),
        avg_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
    )


def benchmark_restrict_logits(
    processor: Any, vocab_size: int, num_iterations: int = 1000
) -> BenchmarkResult:
    """Benchmark logits restriction time."""
    times = []
    logits = torch.randn(vocab_size, device="cpu")
    allowed_ids = list(range(100, 200))  # 100 allowed tokens

    for _ in range(num_iterations):
        # Reset logits
        logits_copy = logits.clone()

        start = time.perf_counter()
        processor._restrict_logits_to_ids(logits_copy, allowed_ids)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        num_calls=num_iterations,
        total_time_ms=sum(times),
        avg_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
    )


def benchmark_apply_empty(
    processor: Any, batch_size: int, vocab_size: int, num_iterations: int = 1000
) -> BenchmarkResult:
    """Benchmark apply() with no active requests (baseline)."""
    times = []
    logits = torch.randn(batch_size, vocab_size, device="cpu")

    # Ensure no active requests
    processor.req_states = {}

    for _ in range(num_iterations):
        logits_copy = logits.clone()

        start = time.perf_counter()
        _ = processor.apply(logits_copy)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        num_calls=num_iterations,
        total_time_ms=sum(times),
        avg_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
    )


def benchmark_full_apply(
    processor: Any, batch_size: int, vocab_size: int, valid_moves: dict, num_iterations: int = 100
) -> BenchmarkResult:
    """
    Benchmark apply() with active constraint requests.
    This simulates the real-world case where the processor is actively constraining outputs.
    """
    from src.inference.logits import RequestState

    times = []
    logits = torch.randn(batch_size, vocab_size, device="cpu")

    for _iteration in range(num_iterations):
        # Set up active request states
        processor.req_states = {}
        for idx in range(batch_size):
            # Build trie for this request
            root = processor._build_trie(valid_moves)

            # Simulate partial generation with <orders> tag detected
            # Generate some tokens as if we're mid-generation
            sample_output = "<orders>\nA PAR - BUR\n"
            output_token_ids = processor.tokenizer.encode(sample_output, add_special_tokens=False)

            state = RequestState(
                root=root,
                output_token_ids=MockOutputTokenIds(output_token_ids).token_ids,  # pyright: ignore[reportArgumentType]
                eos_token_id=processor.eos_token_id,
                newline_token_id=processor.newline_token_id,
                valid_moves_dict=valid_moves,
                expected_orders=len(valid_moves),
            )
            # Mark as active
            state.in_orders = True
            state.current_node = root
            state.last_processed_len = 0

            processor.req_states[idx] = state

        logits_copy = logits.clone()

        start = time.perf_counter()
        _ = processor.apply(logits_copy)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    return BenchmarkResult(
        num_calls=num_iterations,
        total_time_ms=sum(times),
        avg_time_ms=sum(times) / len(times),
        min_time_ms=min(times),
        max_time_ms=max(times),
    )


def main():
    print("=" * 60)
    print("DiplomacyLogitsProcessor Benchmark")
    print("=" * 60)

    # Import and setup
    from transformers import AutoTokenizer

    from src.inference.logits import DiplomacyLogitsProcessor

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create a mock config for the processor
    class MockModelConfig:
        model = model_name

    class MockVllmConfig:
        model_config = MockModelConfig()

    print("Initializing DiplomacyLogitsProcessor...")
    processor = DiplomacyLogitsProcessor(
        vllm_config=MockVllmConfig(),  # pyright: ignore[reportArgumentType]
        device=torch.device("cpu"),
        is_pin_memory=False,
    )

    # Get realistic valid moves
    valid_moves = get_realistic_valid_moves()
    print(f"Using {len(valid_moves)} units with valid moves")

    # Vocab size for Qwen
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    # Benchmark parameters
    batch_sizes = [1, 8, 32, 64]

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    # 1. Trie build time
    trie_result = benchmark_trie_build(processor, valid_moves, num_iterations=100)
    trie_result.print_report("Trie Build (per valid_moves_dict)")

    # 2. Logits restriction time
    restrict_result = benchmark_restrict_logits(processor, vocab_size, num_iterations=1000)
    restrict_result.print_report("Logits Restriction (100 allowed tokens)")

    # 3. Apply with empty state (baseline)
    for batch_size in batch_sizes:
        empty_result = benchmark_apply_empty(processor, batch_size, vocab_size, num_iterations=1000)
        empty_result.print_report(f"Apply (empty, batch={batch_size})")

    # 4. Apply with active constraints
    print("\n" + "-" * 40)
    print("Active Constraint Benchmarks (realistic)")
    print("-" * 40)

    for batch_size in batch_sizes:
        active_result = benchmark_full_apply(
            processor, batch_size, vocab_size, valid_moves, num_iterations=50
        )
        active_result.print_report(f"Apply (active, batch={batch_size})")

    # Summary and comparison
    print("\n" + "=" * 60)
    print("OVERHEAD ANALYSIS")
    print("=" * 60)

    # Compare to expected decode time
    # vLLM on A100 typically does ~500-2000 output tokens/sec for 7B model
    # At 1000 tokens/sec, that's 1ms per token
    # A batch of 32 requests with ~30 tokens each = 960 tokens
    # Expected decode time: ~960ms for the batch (without logits processor)

    typical_batch = 32
    tokens_per_request = 30
    output_tps = 1000  # Conservative estimate
    expected_decode_time_ms = (typical_batch * tokens_per_request) / output_tps * 1000

    # Per-token overhead from logits processor
    active_32 = benchmark_full_apply(processor, 32, vocab_size, valid_moves, num_iterations=50)
    per_apply_overhead_ms = active_32.avg_time_ms
    num_applies = tokens_per_request  # One apply per token
    total_overhead_ms = per_apply_overhead_ms * num_applies

    print(f"\nTypical batch scenario (batch={typical_batch}, {tokens_per_request} tokens/request):")
    print(f"  Expected decode time: ~{expected_decode_time_ms:.0f}ms")
    print(f"  Logits processor overhead: ~{total_overhead_ms:.1f}ms")
    print(f"  Overhead percentage: {total_overhead_ms / expected_decode_time_ms * 100:.1f}%")

    if total_overhead_ms / expected_decode_time_ms < 0.05:
        print("\n✅ Logits processor overhead is ACCEPTABLE (<5%)")
        print("   Focus optimization efforts elsewhere (scaling, batching)")
    else:
        print("\n⚠️  Logits processor overhead is SIGNIFICANT (≥5%)")
        print("   Consider optimizing trie operations or logits masking")


if __name__ == "__main__":
    main()
