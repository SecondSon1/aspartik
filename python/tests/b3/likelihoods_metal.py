import pytest
from utils.likelihood import load_likelihood_from_fasta

import sys

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood, Likelihood, MetalLikelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY, JC, Substiution4
from aspartik.b3.utils.analytical import analytical_hky_2leaf, analytical_jc_2leaf
from aspartik.data import DNASeq
from aspartik.data.msa import MSA
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

_metal_available = MetalLikelihood is not None and sys.platform == "darwin"

pytestmark = pytest.mark.skipif(
    not _metal_available,
    reason="MetalLikelihood not available on this platform",
)


def _load_metal(
    rng: RNG, substitution: Substiution4, scale_ln: int = 30
) -> tuple[Tree, Likelihood]:
    _, tree, ll = load_likelihood_from_fasta(
        "data/alignments/apes.fasta",
        rng=rng,
        substitution=substitution,
        clock=Clock.Strict(Real(1.0)),
        backend=MetalLikelihood,
        scale_ln=scale_ln,
    )
    return tree, ll


def _metal_cpu_pair(
    rng: RNG, substitution: Substiution4, scale_ln: int = 30
) -> tuple[Tree, MetalLikelihood, CPU4Likelihood]:
    msa = read_msa_from_fasta("data/alignments/apes.fasta")
    tree = Tree(msa.sequence_names(), rng)
    clock_rate = Real(1.0)
    clock = Clock.Strict(clock_rate)

    metal_ll = MetalLikelihood(
        msa=msa,
        substitution=substitution,
        clock=clock,
        tree=tree,
        scale_ln=scale_ln,
    )
    cpu_ll = CPU4Likelihood(
        msa=msa,
        substitution=substitution,
        clock=clock,
        tree=tree,
        scale_ln=scale_ln,
    )
    return tree, metal_ll, cpu_ll


class TestMetalLifecycle:
    def test_num_patterns(self, rng: RNG) -> None:
        _, ll = _load_metal(rng, JC())
        assert ll.num_patterns() == 69

    def test_fuzz(self, rng: RNG) -> None:
        _, ll = _load_metal(rng, HKY(RealVector(0.1, 0.2, 0.3, 0.4), Real(2.0)))

        for _ in range(1000):
            match rng.random_int(0, 3):
                case 0:
                    _ = ll.likelihood()
                case 1:
                    ll.accept()
                case 2:
                    ll.reject()

    def test_reject_restores_exact_value(self, rng: RNG) -> None:
        tree, ll = _load_metal(rng, HKY(RealVector(0.25, 0.25, 0.25, 0.25), Real(2.0)))

        l1 = ll.likelihood()
        ll.accept()
        tree.accept()

        tree.set_height(tree.root, tree.height_of(tree.root) * 1.5)
        l2 = ll.likelihood()
        assert l1 != l2

        ll.reject()
        tree.reject()

        l3 = ll.likelihood()
        ll.accept()
        tree.accept()
        assert l3 == l1, f"reject should restore exactly: got {l3}, expected {l1}"

    def test_multiple_accept_reject_cycles(self, rng: RNG) -> None:
        tree, ll = _load_metal(rng, JC())

        l_base = ll.likelihood()
        ll.accept()
        tree.accept()

        for i in range(50):
            root = tree.root
            old_height = tree.height_of(root)
            tree.set_height(root, old_height * (1.0 + 0.01 * (i % 5)))

            l_new = ll.likelihood()

            if i % 3 == 0:
                ll.accept()
                tree.accept()
                l_base = l_new
            else:
                ll.reject()
                tree.reject()
                l_restored = ll.likelihood()
                ll.accept()
                tree.accept()
                assert l_restored == l_base, (
                    f"cycle {i}: reject did not restore: "
                    f"got {l_restored}, expected {l_base}"
                )


_SCALE_LNS = [3, 30, 300]
_REL_TOL = 1e-7


class TestMetalMatchesCpu:
    @pytest.mark.parametrize("scale_ln", _SCALE_LNS)
    def test_initial_likelihood_matches_cpu(self, rng: RNG, scale_ln: int) -> None:
        _, metal_ll, cpu_ll = _metal_cpu_pair(
            rng, HKY(RealVector(0.25, 0.25, 0.25, 0.25), Real(2.0)), scale_ln
        )

        l_metal = metal_ll.likelihood()
        l_cpu = cpu_ll.likelihood()

        assert abs(l_metal - l_cpu) <= _REL_TOL * abs(l_cpu), (
            f"scale_ln={scale_ln}: Metal={l_metal}, CPU={l_cpu}"
        )

    @pytest.mark.parametrize("scale_ln", _SCALE_LNS)
    def test_after_tree_change_matches_cpu(self, rng: RNG, scale_ln: int) -> None:
        tree, metal_ll, cpu_ll = _metal_cpu_pair(
            rng, HKY(RealVector(0.3, 0.2, 0.2, 0.3), Real(5.0)), scale_ln
        )

        # Accept the initial state
        metal_ll.accept()
        cpu_ll.accept()
        tree.accept()

        # Modify the tree height
        tree.set_height(tree.root, tree.height_of(tree.root) * 2.0)

        l_metal = metal_ll.likelihood()
        l_cpu = cpu_ll.likelihood()

        assert abs(l_metal - l_cpu) <= _REL_TOL * abs(l_cpu), (
            f"scale_ln={scale_ln}: Metal={l_metal}, CPU={l_cpu}"
        )


# ─── Analytical tests ─────────────────────────────────────────────────────────

# f32 GPU vs f64 analytical reference; single-precision gives ~1e-7 absolute
# error for typical likelihood values (-10 to -500 range).
_EPS = 1e-4


class TestMetalAnalytical:
    @pytest.mark.parametrize(
        ("seq1", "seq2", "height", "clock_rate"),
        (
            pytest.param("A", "A", 1.0, 1.0, id="single_identical"),
            pytest.param("ACGTACGTACGT", "ACGTACGTACGT", 0.5, 1.0, id="identical"),
            pytest.param("AAACCCGGGTTT", "ACGTACGTACGT", 0.3, 1.5, id="different"),
            pytest.param("ACGT", "TGCA", 0.1, 3.0, id="all_different"),
        ),
    )
    def test_jc_2leaf(
        self,
        seq1: str,
        seq2: str,
        height: float,
        clock_rate: float,
        rng: RNG,
    ) -> None:
        tree = Tree(["A", "B"], rng)
        tree.set_height(tree.root, height)
        msa = MSA(["A", "B"], [DNASeq(seq1), DNASeq(seq2)])

        ll = MetalLikelihood(
            msa=msa,
            substitution=JC(),
            clock=Clock.Strict(Real(clock_rate)),
            tree=tree,
        )

        result = ll.likelihood()
        expected = analytical_jc_2leaf(seq1, seq2, clock_rate * height)

        assert abs(result - expected) < _EPS, f"Metal={result}, expected={expected}"

    @pytest.mark.parametrize(
        ("seq1", "seq2", "height", "kappa", "freqs"),
        (
            pytest.param(
                "AACCGGTTACGT",
                "ACGTACGTACGT",
                0.4,
                2.0,
                (0.25, 0.25, 0.25, 0.25),
                id="equal_freqs",
            ),
            pytest.param(
                "AAACCCGGGTTTACG",
                "ACGTACGTACGTACG",
                0.25,
                4.0,
                (0.3, 0.2, 0.2, 0.3),
                id="unequal_freqs",
            ),
        ),
    )
    def test_hky_2leaf(
        self,
        seq1: str,
        seq2: str,
        height: float,
        kappa: float,
        freqs: tuple[float, ...],
        rng: RNG,
    ) -> None:
        tree = Tree(["A", "B"], rng)
        tree.set_height(tree.root, height)
        msa = MSA(["A", "B"], [DNASeq(seq1), DNASeq(seq2)])

        ll = MetalLikelihood(
            msa=msa,
            substitution=HKY(RealVector(*freqs), Real(kappa)),
            clock=Clock.Strict(Real(1.0)),
            tree=tree,
        )

        result = ll.likelihood()
        expected = analytical_hky_2leaf(seq1, seq2, height, kappa, freqs)

        assert abs(result - expected) < _EPS, f"Metal={result}, expected={expected}"
