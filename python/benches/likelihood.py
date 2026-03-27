"""
Likelihood benchmarks for Felsenstein's pruning algorithm.

Two modes:
  1. Isolated: tight loop of root-height perturbation + likelihood eval.
     Measures best-case single-path recomputation.
  2. MCMC: full MCMC run with tree operators.  Measures realistic per-step
     likelihood cost including subtreeLeap, SPR, etc.

BEAST1 comparison uses mode 2.  JVM startup and BEAGLE initialization are
excluded: BEAST1 runs a long chain and only its own reported timer is used.

Usage:
    python python/benches/likelihood.py data/alignments/influenza.fasta
    python python/benches/likelihood.py data/alignments/influenza.fasta --beast1
    python python/benches/likelihood.py data/alignments/respiratory.fasta --iters 500
"""

import argparse
import re
import subprocess
import tempfile
import time

from aspartik.b3 import Clock
from aspartik.b3.config import b3_config, beast1_config
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.data.msa import MSA
from aspartik.io import FastaReader
from aspartik.rng import RNG


def bench_b3_isolated(
    msa: MSA,
    iters: int,
    warmup: int,
    num_threads: int,
) -> float:
    """Best-case: only root height changes, single dirty path."""
    rng = RNG(4)
    tree = Tree(msa.sequence_names(), rng)

    ll = CPU4Likelihood(
        msa=msa,
        substitution=HKY(RealVector(0.25, 0.25, 0.25, 0.25), Real(2.0)),
        clock=Clock.Strict(Real(1.0)),
        tree=tree,
        num_threads=num_threads,
    )

    for i in range(warmup):
        tree.set_height(tree.root, tree.height_of(tree.root) * (1.0 + 0.001 * (i % 5)))
        ll.likelihood()
        ll.accept()
        tree.accept()

    start = time.perf_counter()
    for i in range(iters):
        tree.set_height(tree.root, tree.height_of(tree.root) * (1.0 + 0.001 * (i % 5)))
        ll.likelihood()
        ll.accept()
        tree.accept()
    end = time.perf_counter()

    return (end - start) / iters


def bench_b3_mcmc(msa: MSA, length: int) -> float:
    """Realistic: full MCMC with tree operators, reports total time."""
    mcmc = b3_config(
        msa,
        tree_prior="constant",
        substitution_model="HKY",
        print_every=None,
    )

    start = time.perf_counter()
    mcmc.run(length)
    end = time.perf_counter()

    return (end - start) / length


def bench_beast1(msa: MSA, length: int) -> float:
    """
    Run BEAST1 MCMC and extract per-step time from its own timer.

    Uses a long chain to amortize JVM startup / BEAGLE init.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = f"{tmpdir}/beast.log"

        config = beast1_config(
            msa,
            tree_prior="constant",
            substitution_model="HKY",
            screen_log_every=None,
            log_path=log_path,
            length=length,
        )

        xml_path = f"{tmpdir}/beast.xml"
        with open(xml_path, "w") as f:
            f.write(config)

        result = subprocess.run(
            ["beast", "-seed", "4", "-citations_off", "-overwrite", "-beagle_CPU", xml_path],
            capture_output=True,
            text=True,
            timeout=600,
        )

    if result.returncode != 0:
        raise RuntimeError(f"BEAST1 failed:\n{result.stderr}")

    # BEAST1 prints: "Total calculation time: X.XXX seconds"
    m = re.search(r"Total calculation time:\s+([\d.]+)\s+seconds", result.stdout)
    if not m:
        raise RuntimeError(
            f"Could not parse BEAST1 timer from output:\n{result.stdout[-500:]}"
        )

    total_seconds = float(m.group(1))
    return total_seconds / length


def format_time(seconds_per_eval: float) -> str:
    us = seconds_per_eval * 1e6
    if us < 1000:
        return f"{us:.1f} μs/step"
    ms = us / 1000
    if ms < 1000:
        return f"{ms:.2f} ms/step"
    return f"{seconds_per_eval:.3f} s/step"


def main():
    parser = argparse.ArgumentParser(description="Likelihood benchmark")
    parser.add_argument("fasta_path", type=str)
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument("--iters", type=int, default=5000, help="isolated-mode iterations")
    parser.add_argument("--warmup", type=int, default=100, help="isolated-mode warmup")
    parser.add_argument("--num_threads", type=int, default=0, help="b3 thread count (0=auto)")
    parser.add_argument("--length", type=int, default=100_000, help="MCMC chain length for b3 and BEAST1")
    parser.add_argument("--beast1", action="store_true", help="also benchmark BEAST1+BEAGLE")
    args = parser.parse_args()

    records = list(FastaReader.from_file(args.fasta_path))
    if args.num_sequences:
        records = records[: args.num_sequences]
    msa = MSA.from_fasta(records)

    print(f"Alignment: {msa.num_sequences} seqs x {msa.num_sites} sites ({msa.num_sequences * msa.num_sites:,} cells)")
    print(f"Model:     HKY + strict clock")
    print()

    # --- isolated benchmark ---
    isolated_time = bench_b3_isolated(msa, args.iters, args.warmup, args.num_threads)
    print(f"b3 isolated:  {format_time(isolated_time):>15}  (root-height only, {args.iters} iters)")

    # --- MCMC benchmark ---
    mcmc_time = bench_b3_mcmc(msa, args.length)
    print(f"b3 MCMC:      {format_time(mcmc_time):>15}  ({args.length} steps)")

    if args.beast1:
        beast1_time = bench_beast1(msa, args.length)
        print(f"BEAST1 MCMC:  {format_time(beast1_time):>15}  ({args.length} steps, JVM startup excluded)")
        print()
        ratio = beast1_time / mcmc_time
        print(f"MCMC ratio:   b3 is {ratio:.1f}x {'faster' if ratio > 1 else 'slower'}")


if __name__ == "__main__":
    main()
