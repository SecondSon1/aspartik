import pytest
from utils.likelihood import load_likelihood_from_fasta

from collections.abc import Callable
from typing import Any

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood, CUDALikelihood, Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY, JC, Substiution4
from aspartik.rng import RNG


def _load(
    rng: RNG,
    substitution: Substiution4,
    ll_backend: Callable[..., Likelihood],
) -> tuple[Tree, Likelihood]:
    _, tree, ll = load_likelihood_from_fasta(
        "data/alignments/apes.fasta",
        rng=rng,
        substitution=substitution,
        clock=Clock.Strict(Real(1.0)),
        backend=ll_backend,
    )
    return tree, ll


def _cuda_detected() -> bool:
    try:
        _load(RNG(5), JC(), CUDALikelihood)
        return True
    except Exception:
        return False


def _likelihood_backends() -> list[Any]:
    params = [pytest.param(CPU4Likelihood, id="cpu")]
    if _cuda_detected():
        params.append(pytest.param(CUDALikelihood, id="cuda"))
    return params


_LIKELIHOOD_BACKENDS = _likelihood_backends()


class TestLikelihoods:
    @pytest.mark.parametrize(argnames=["ll_backend"], argvalues=_LIKELIHOOD_BACKENDS)
    def test_fuzz(self, rng: RNG, ll_backend: Callable[..., Likelihood]) -> None:
        _, ll = _load(rng, HKY(RealVector(0.1, 0.2, 0.3, 0.4), Real(2.0)), ll_backend)

        assert ll.num_patterns() == 69

        for _ in range(1000):
            match rng.random_int(0, 3):
                case 0:
                    ll.likelihood()
                case 1:
                    ll.accept()
                case 2:
                    ll.reject()

    @pytest.mark.parametrize(argnames=["ll_backend"], argvalues=_LIKELIHOOD_BACKENDS)
    def test_reject_restores_exact_value(
        self, rng: RNG, ll_backend: Callable[..., Likelihood]
    ) -> None:
        tree, ll = _load(
            rng, HKY(RealVector(0.25, 0.25, 0.25, 0.25), Real(2.0)), ll_backend
        )
        l1 = ll.likelihood()
        ll.accept()
        tree.accept()

        tree.set_height(tree.root, tree.height_of(tree.root) * 1.5)
        l2 = ll.likelihood()
        assert l1 != l2, "set height altered tree, likelihood should be recomputed"

        ll.reject()
        tree.reject()

        l3 = ll.likelihood()
        ll.accept()
        tree.accept()
        assert l3 == l1, f"reject should restore exactly: got {l3}, expected {l1}"

    @pytest.mark.parametrize(argnames=["ll_backend"], argvalues=_LIKELIHOOD_BACKENDS)
    def test_multiple_accept_reject_cycles(
        self, rng: RNG, ll_backend: Callable[..., Likelihood]
    ) -> None:
        tree, ll = _load(rng, JC(), ll_backend)

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
