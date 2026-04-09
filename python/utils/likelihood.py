import pytest
from _pytest.mark.structures import ParameterSet

from collections.abc import Callable
from typing import Any

from aspartik._aspartik_rust_impl._b3_rust_impl import Clock
from aspartik.b3.likelihoods import (
    CPU4Likelihood,
    CUDALikelihood,
    Likelihood,
    MetalLikelihood,
)
from aspartik.b3.parameters import Real, Tree
from aspartik.b3.substitutions import JC, Substiution4
from aspartik.data import DNASeq
from aspartik.data.msa import MSA
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


def _probe_backend(backend: Callable[..., Likelihood]) -> bool:
    msa = MSA(["A", "B"], [DNASeq("ACGT"), DNASeq("ACGT")])
    tree = Tree(["A", "B"], RNG(0))
    try:
        backend(msa=msa, substitution=JC(), clock=Clock.Strict(Real(1.0)), tree=tree)
        return True
    except Exception:
        return False


def cuda_available() -> bool:
    return _probe_backend(CUDALikelihood)


def metal_available() -> bool:
    if MetalLikelihood is None:
        return False
    return _probe_backend(MetalLikelihood)


def likelihood_backend_params() -> list[ParameterSet]:
    params = [pytest.param(CPU4Likelihood, id="cpu")]
    if cuda_available():
        params.append(pytest.param(CUDALikelihood, id="cuda"))
    if metal_available():
        params.append(pytest.param(MetalLikelihood, id="metal"))
    return params


LIKELIHOOD_BACKEND_PARAMS: list[ParameterSet] = likelihood_backend_params()


def load_likelihood_from_fasta(
    fasta_path: str,
    *,
    rng: RNG,
    substitution: Substiution4,
    clock: Clock,
    backend: Callable[..., Likelihood],
    **kwargs: Any,
) -> tuple[MSA, Tree, Likelihood]:
    msa = read_msa_from_fasta(fasta_path)
    tree = Tree(msa.sequence_names(), rng)
    ll = backend(msa=msa, substitution=substitution, clock=clock, tree=tree, **kwargs)
    return msa, tree, ll
