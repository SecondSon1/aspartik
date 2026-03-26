"""
BEAST1 cross-reference tests for Felsenstein's pruning algorithm.

Each test builds a b3 likelihood on a fixed tree, exports the tree as Newick,
runs BEAST1 with the same tree and model parameters for zero MCMC steps, and
compares the initial treeLikelihood value.

Requires BEAST1 and BEAGLE installed.  Skipped by default; run with:
    pytest -m likelihood      (apes + electricFish, ~2s)
    pytest -m likelihood_slow (influenza, ~5s)
"""

import pytest

from aspartik.b3 import Clock
from aspartik.b3.config._beast1 import beast1_likelihood
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


def _compare_beast1(
    fasta_path: str,
    kappa_val: float,
    freqs: tuple[float, ...],
    clock_rate_val: float,
    seed: int = 4,
    tolerance: float = 1e-10,
):
    rng = RNG(seed)
    msa = read_msa_from_fasta(fasta_path)
    names = list(msa.sequence_names())

    tree = Tree(names, rng)

    ll = CPU4Likelihood(
        msa=msa,
        substitution=HKY(RealVector(*freqs), Real(kappa_val)),
        clock=Clock.Strict(Real(clock_rate_val)),
        tree=tree,
    )

    b3_ll = ll.likelihood()
    newick = tree.to_newick()
    sequences = [str(msa.sequence(i)) for i in range(msa.num_sequences)]

    beast1_ll = beast1_likelihood(
        names,
        sequences,
        newick,
        kappa_val,
        freqs,
        clock_rate_val,
    )

    diff = abs(b3_ll - beast1_ll)
    rel = diff / abs(beast1_ll)
    assert rel < tolerance, f"b3={b3_ll}, beast1={beast1_ll}, diff={diff}, rel={rel}"


@pytest.mark.likelihood
class TestApes:
    def test_hky_equal_freqs(self):
        _compare_beast1(
            "data/alignments/apes.fasta",
            kappa_val=2.0,
            freqs=(0.25, 0.25, 0.25, 0.25),
            clock_rate_val=1.0,
        )

    def test_hky_unequal_freqs(self):
        _compare_beast1(
            "data/alignments/apes.fasta",
            kappa_val=5.0,
            freqs=(0.3, 0.2, 0.2, 0.3),
            clock_rate_val=0.5,
        )


@pytest.mark.likelihood
class TestElectricFish:
    def test_hky(self):
        _compare_beast1(
            "data/alignments/electricFish.fasta",
            kappa_val=3.0,
            freqs=(0.25, 0.25, 0.25, 0.25),
            clock_rate_val=1.0,
        )

    def test_hky_unequal_freqs(self):
        _compare_beast1(
            "data/alignments/electricFish.fasta",
            kappa_val=6.0,
            freqs=(0.35, 0.15, 0.2, 0.3),
            clock_rate_val=0.8,
        )


@pytest.mark.likelihood
class TestInfluenza:
    def test_hky(self):
        _compare_beast1(
            "data/alignments/influenza.fasta",
            kappa_val=2.0,
            freqs=(0.25, 0.25, 0.25, 0.25),
            clock_rate_val=1.0,
        )
