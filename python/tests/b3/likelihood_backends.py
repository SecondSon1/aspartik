import pytest
from _pytest.mark.structures import ParameterSet
from utils.likelihood import cuda_available, metal_available

from collections.abc import Callable
from typing import Any

from aspartik.b3 import Clock
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
    Likelihood,
    MetalLikelihood,
)
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY, Substiution4
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

# Metal only natively supports f32, thus precision is lower
_REL_TOL_F64 = 1e-10
_REL_TOL_F32 = 1e-7

_GPU_TEST_PARAMS: list[ParameterSet] = []
if cuda_available():
    _GPU_TEST_PARAMS.append(pytest.param(CUDALikelihood, None, _REL_TOL_F64, id="cuda"))
if metal_available():
    _GPU_TEST_PARAMS += [
        pytest.param(MetalLikelihood, scale_ln, _REL_TOL_F32, id=f"metal-{scale_ln}")
        for scale_ln in [3, 20, 60]
    ]

if not _GPU_TEST_PARAMS:
    pytest.skip("no GPU backend available", allow_module_level=True)


def _gpu_cpu_pair(
    rng: RNG,
    substitution: Substiution4,
    gpu_backend: Callable[..., Likelihood],
    scale_ln: int | None,
) -> tuple[Tree, Likelihood, CPU4Likelihood]:
    msa = read_msa_from_fasta("data/alignments/apes.fasta")
    tree = Tree(msa.sequence_names(), rng)
    kwargs: dict[str, Any] = dict(
        msa=msa,
        substitution=substitution,
        clock=Clock.Strict(Real(1.0)),
        tree=tree,
    )
    if scale_ln is not None:
        kwargs["scale_ln"] = scale_ln

    gpu_ll = gpu_backend(**kwargs)
    cpu_ll = CPU4Likelihood(**kwargs)
    return tree, gpu_ll, cpu_ll


class TestGpuMatchesCpu:
    @pytest.mark.parametrize(("gpu_backend", "scale_ln", "rel_tol"), _GPU_TEST_PARAMS)
    def test_initial_likelihood_matches_cpu(
        self,
        rng: RNG,
        gpu_backend: Callable[..., Likelihood],
        scale_ln: int | None,
        rel_tol: float,
    ) -> None:
        _, gpu_ll, cpu_ll = _gpu_cpu_pair(
            rng,
            HKY(RealVector(0.25, 0.25, 0.25, 0.25), Real(2.0)),
            gpu_backend,
            scale_ln,
        )

        l_gpu = gpu_ll.likelihood()
        l_cpu = cpu_ll.likelihood()

        assert abs(l_gpu - l_cpu) <= rel_tol * abs(l_cpu), (
            f"scale_ln={scale_ln}: GPU={l_gpu}, CPU={l_cpu}"
        )

    @pytest.mark.parametrize(("gpu_backend", "scale_ln", "rel_tol"), _GPU_TEST_PARAMS)
    def test_after_tree_change_matches_cpu(
        self,
        rng: RNG,
        gpu_backend: Callable[..., Likelihood],
        scale_ln: int | None,
        rel_tol: float,
    ) -> None:
        tree, gpu_ll, cpu_ll = _gpu_cpu_pair(
            rng, HKY(RealVector(0.3, 0.2, 0.2, 0.3), Real(5.0)), gpu_backend, scale_ln
        )

        gpu_ll.accept()
        cpu_ll.accept()
        tree.accept()

        tree.set_height(tree.root, tree.height_of(tree.root) * 2.0)

        l_gpu = gpu_ll.likelihood()
        l_cpu = cpu_ll.likelihood()

        assert abs(l_gpu - l_cpu) <= rel_tol * abs(l_cpu), (
            f"scale_ln={scale_ln}: GPU={l_gpu}, CPU={l_cpu}"
        )
