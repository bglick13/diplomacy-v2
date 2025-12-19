#!/usr/bin/env python3
"""
Benchmark: Sequential vs Concurrent adapter processing in vLLM.

This script compares two approaches for handling mixed-adapter batches:
1. Sequential: Process each adapter group one at a time
2. Concurrent: Process all prompts at once, let vLLM handle multi-adapter batching

Usage:
    python scripts/benchmark_adapter_batching.py
    python scripts/benchmark_adapter_batching.py --num-prompts 42 --iterations 5
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass

import modal


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    approach: str
    num_prompts: int
    num_adapters: int
    wall_time_s: float
    generation_time_s: float
    total_tokens: int
    tokens_per_second: float

    def __str__(self) -> str:
        return (
            f"{self.approach:12} | {self.num_prompts:3} prompts | "
            f"{self.num_adapters} adapters | {self.wall_time_s:6.2f}s wall | "
            f"{self.generation_time_s:6.2f}s gen | {self.total_tokens:4} tokens | "
            f"{self.tokens_per_second:6.1f} tok/s"
        )


def get_test_prompts(num_prompts: int) -> list[str]:
    """Generate realistic Diplomacy prompts."""
    base_prompt = """You are playing as FRANCE in a game of Diplomacy.

Current phase: SPRING 1901 MOVEMENT

Your units:
- A PAR (Army in Paris)
- A MAR (Army in Marseilles)
- F BRE (Fleet in Brest)

Valid moves for your units are provided. Analyze the situation and issue orders.

<orders>
"""
    return [base_prompt] * num_prompts


def get_test_valid_moves(num_prompts: int) -> list[dict]:
    """Generate valid moves for each prompt."""
    moves = {
        "A PAR": ["A PAR H", "A PAR - BUR", "A PAR - PIC", "A PAR - GAS"],
        "A MAR": ["A MAR H", "A MAR - BUR", "A MAR - PIE", "A MAR - SPA"],
        "F BRE": ["F BRE H", "F BRE - MAO", "F BRE - ENG", "F BRE - PIC"],
    }
    return [moves] * num_prompts


def get_mixed_adapter_names(num_prompts: int, adapters: list[str]) -> list[str]:
    """Assign adapters round-robin to prompts."""
    return [adapters[i % len(adapters)] for i in range(num_prompts)]


async def benchmark_concurrent(
    engine,
    prompts: list[str],
    valid_moves: list[dict],
    lora_names: list[str],
) -> BenchmarkResult:
    """Benchmark concurrent processing (current approach)."""
    start = time.time()

    results = await engine.generate.remote.aio(
        prompts=prompts,
        valid_moves=valid_moves,
        lora_names=lora_names,
        temperature=0.8,
        max_new_tokens=128,
    )

    wall_time = time.time() - start
    total_tokens = sum(len(r.get("token_ids", [])) for r in results)

    return BenchmarkResult(
        approach="concurrent",
        num_prompts=len(prompts),
        num_adapters=len(set(lora_names)),
        wall_time_s=wall_time,
        generation_time_s=wall_time,  # Can't separate without internal timing
        total_tokens=total_tokens,
        tokens_per_second=total_tokens / wall_time if wall_time > 0 else 0,
    )


async def benchmark_sequential(
    engine,
    prompts: list[str],
    valid_moves: list[dict],
    lora_names: list[str],
) -> BenchmarkResult:
    """Benchmark sequential processing (grouped by adapter)."""
    from collections import defaultdict

    # Group by adapter
    adapter_groups: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    for i, (prompt, moves, adapter) in enumerate(
        zip(prompts, valid_moves, lora_names, strict=False)
    ):
        adapter_groups[adapter].append((i, prompt, moves))

    start = time.time()
    all_results: list[tuple[int, dict]] = []

    # Process each adapter group sequentially
    for adapter, group in adapter_groups.items():
        indices = [g[0] for g in group]
        group_prompts = [g[1] for g in group]
        group_moves = [g[2] for g in group]
        group_adapters = [adapter] * len(group)

        results = await engine.generate.remote.aio(
            prompts=group_prompts,
            valid_moves=group_moves,
            lora_names=group_adapters,
            temperature=0.8,
            max_new_tokens=128,
        )

        for idx, result in zip(indices, results, strict=False):
            all_results.append((idx, result))

    wall_time = time.time() - start

    # Sort by original index
    all_results.sort(key=lambda x: x[0])
    results = [r[1] for r in all_results]

    total_tokens = sum(len(r.get("token_ids", [])) for r in results)

    return BenchmarkResult(
        approach="sequential",
        num_prompts=len(prompts),
        num_adapters=len(set(lora_names)),
        wall_time_s=wall_time,
        generation_time_s=wall_time,
        total_tokens=total_tokens,
        tokens_per_second=total_tokens / wall_time if wall_time > 0 else 0,
    )


async def run_benchmarks(
    num_prompts: int,
    adapters: list[str],
    iterations: int,
    model_id: str,
) -> None:
    """Run benchmark suite."""
    print("=" * 80)
    print("Adapter Batching Benchmark: Sequential vs Concurrent")
    print("=" * 80)
    print(f"Prompts per batch: {num_prompts}")
    print(f"Adapters: {adapters}")
    print(f"Iterations: {iterations}")
    print()

    # Get inference engine
    InferenceEngine = modal.Cls.from_name("diplomacy-grpo", "InferenceEngine")
    engine = InferenceEngine(model_id=model_id)

    # Warmup - run multiple times to ensure caches are hot
    print("🔥 Warming up (3 rounds to fill caches)...")
    warmup_prompts = get_test_prompts(num_prompts)
    warmup_moves = get_test_valid_moves(num_prompts)
    warmup_adapters = get_mixed_adapter_names(num_prompts, adapters)

    for i in range(3):
        await engine.generate.remote.aio(
            prompts=warmup_prompts,
            valid_moves=warmup_moves,
            lora_names=warmup_adapters,
            temperature=0.8,
            max_new_tokens=128,
        )
        print(f"  Warmup {i + 1}/3 complete")
    print("✅ Warmup complete - caches should be hot\n")

    # Prepare test data
    prompts = get_test_prompts(num_prompts)
    valid_moves = get_test_valid_moves(num_prompts)
    lora_names = get_mixed_adapter_names(num_prompts, adapters)

    print(f"Adapter distribution: { {a: lora_names.count(a) for a in set(lora_names)} }")
    print()

    # Run benchmarks
    concurrent_results: list[BenchmarkResult] = []
    sequential_results: list[BenchmarkResult] = []

    for i in range(iterations):
        print(f"--- Iteration {i + 1}/{iterations} ---")

        # Concurrent
        result = await benchmark_concurrent(engine, prompts, valid_moves, lora_names)
        concurrent_results.append(result)
        print(f"  {result}")

        # Sequential
        result = await benchmark_sequential(engine, prompts, valid_moves, lora_names)
        sequential_results.append(result)
        print(f"  {result}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    def avg(results: list[BenchmarkResult], field: str) -> float:
        return sum(getattr(r, field) for r in results) / len(results)

    print(f"\nConcurrent (avg over {iterations} runs):")
    print(f"  Wall time:     {avg(concurrent_results, 'wall_time_s'):.2f}s")
    print(f"  Tokens/sec:    {avg(concurrent_results, 'tokens_per_second'):.1f}")

    print(f"\nSequential (avg over {iterations} runs):")
    print(f"  Wall time:     {avg(sequential_results, 'wall_time_s'):.2f}s")
    print(f"  Tokens/sec:    {avg(sequential_results, 'tokens_per_second'):.1f}")

    speedup = avg(sequential_results, "wall_time_s") / avg(concurrent_results, "wall_time_s")
    print(
        f"\n🏁 Concurrent is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than sequential"
    )

    # Recommendation
    print("\n" + "=" * 80)
    if speedup > 1.2:
        print("✅ RECOMMENDATION: Use CONCURRENT (current implementation)")
        print("   The latency improvement outweighs any throughput loss from mixed adapters.")
    elif speedup < 0.8:
        print("✅ RECOMMENDATION: Use SEQUENTIAL (revert to grouped processing)")
        print("   The throughput gain from uniform batches outweighs the latency cost.")
    else:
        print("⚖️  RECOMMENDATION: Either approach is fine (< 20% difference)")
        print("   Consider other factors like code simplicity.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark adapter batching strategies")
    parser.add_argument("--num-prompts", type=int, default=14, help="Prompts per batch")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument(
        "--adapters",
        type=str,
        default="grpo-20251219-094501/adapter_v0,grpo-20251219-094501/adapter_v6",
        help="Comma-separated adapter names",
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-7B-Instruct")

    args = parser.parse_args()
    adapters = [a.strip() for a in args.adapters.split(",")]

    asyncio.run(
        run_benchmarks(
            num_prompts=args.num_prompts,
            adapters=adapters,
            iterations=args.iterations,
            model_id=args.model_id,
        )
    )


if __name__ == "__main__":
    main()
