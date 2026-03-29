from collections.abc import Callable
from typing import Any

from aspartik._aspartik_rust_impl._b3_rust_impl import Clock
from aspartik.b3.likelihoods import Likelihood
from aspartik.b3.parameters import Tree
from aspartik.b3.substitutions import Substiution4
from aspartik.data.msa import MSA
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


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
