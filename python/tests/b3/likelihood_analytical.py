import pytest
from utils.likelihood import LIKELIHOOD_BACKEND_PARAMS

from collections.abc import Callable

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood, Likelihood, MetalLikelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY, JC
from aspartik.b3.utils.analytical import analytical_hky_2leaf, analytical_jc_2leaf
from aspartik.data import DNASeq
from aspartik.data.msa import MSA
from aspartik.rng import RNG

_EPS_F64 = 1e-14
_EPS_F32 = 1e-5


def _eps(ll_backend: Callable[..., Likelihood]) -> float:
    return _EPS_F32 if ll_backend is MetalLikelihood else _EPS_F64


class TestJC2Leaf:
    @pytest.mark.parametrize("ll_backend", LIKELIHOOD_BACKEND_PARAMS)
    @pytest.mark.parametrize(
        ("seq1", "seq2", "height", "clock"),
        (
            pytest.param("A", "A", 1.0, 1.0, id="single_site"),
            pytest.param("ACGTACGTACGT", "ACGTACGTACGT", 0.5, 1.0, id="identical"),
            pytest.param("AAACCCGGGTTT", "ACGTACGTACGT", 0.3, 1.5, id="different"),
            pytest.param("ACGT", "TGCA", 0.1, 3.0, id="all_different"),
        ),
    )
    def test_jc2(
        self,
        seq1: str,
        seq2: str,
        height: float,
        clock: float,
        ll_backend: Callable[..., Likelihood],
        rng: RNG,
    ) -> None:
        tree = Tree(["A", "B"], rng)
        tree.set_height(tree.root, height)

        msa = MSA(["A", "B"], [DNASeq(seq1), DNASeq(seq2)])

        ll = ll_backend(
            msa=msa,
            substitution=JC(),
            clock=Clock.Strict(Real(clock)),
            tree=tree,
        )

        b3_ll = ll.likelihood()
        expected = analytical_jc_2leaf(seq1, seq2, clock * height)

        assert abs(b3_ll - expected) < _eps(ll_backend), (
            f"b3={b3_ll}, expected={expected}"
        )


_EQUAL_FREQS = ("AACCGGTTACGT", "ACGTACGTACGT", 0.4, 2.0, (0.25, 0.25, 0.25, 0.25))
_UNEQUAL_FREQS = ("AAACCCGGGTTTACG", "ACGTACGTACGTACG", 0.25, 4.0, (0.3, 0.2, 0.2, 0.3))


class TestHKY2Leaf:
    @pytest.mark.parametrize("ll_backend", LIKELIHOOD_BACKEND_PARAMS)
    @pytest.mark.parametrize(
        ("seq1", "seq2", "height", "kappa", "freqs"),
        (
            pytest.param(*_EQUAL_FREQS, id="equal_freqs"),
            pytest.param(*_UNEQUAL_FREQS, id="unequal_freqs"),
        ),
    )
    def test_hky2(
        self,
        seq1: str,
        seq2: str,
        height: float,
        kappa: float,
        freqs: tuple[float, ...],
        ll_backend: Callable[..., Likelihood],
        rng: RNG,
    ) -> None:
        tree = Tree(["A", "B"], rng)
        tree.set_height(tree.root, height)

        msa = MSA(["A", "B"], [DNASeq(seq1), DNASeq(seq2)])

        ll = ll_backend(
            msa=msa,
            substitution=HKY(RealVector(*freqs), Real(kappa)),
            clock=Clock.Strict(Real(1.0)),
            tree=tree,
        )

        b3_ll = ll.likelihood()
        expected = analytical_hky_2leaf(seq1, seq2, height, kappa, freqs)

        assert abs(b3_ll - expected) < _eps(ll_backend), (
            f"b3={b3_ll}, expected={expected}"
        )
