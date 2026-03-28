import pytest

import glob

from aspartik.b3 import Clock
from aspartik.b3.config._beast1 import beast1_likelihood
from aspartik.b3.likelihoods import CPU4Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG

_FASTA_FILES = sorted(glob.glob("data/alignments/*.fasta"))

_HKY_PARAMS = [
    pytest.param(2.0, (0.25, 0.25, 0.25, 0.25), 1.0, id="equal_freqs"),
    pytest.param(5.0, (0.3, 0.2, 0.2, 0.3), 0.5, id="unequal_freqs"),
]


@pytest.mark.likelihood
class TestBeast1:
    @pytest.mark.parametrize("fasta", _FASTA_FILES, ids=lambda p: p.split("/")[-1])
    @pytest.mark.parametrize(("kappa", "freqs", "clock_rate"), _HKY_PARAMS)
    def test_hky(
        self,
        fasta: str,
        kappa: float,
        freqs: tuple[float, ...],
        clock_rate: float,
        rng: RNG
    ):
        msa = read_msa_from_fasta(fasta)
        tree = Tree(list(msa.sequence_names()), rng)

        ll = CPU4Likelihood(
            msa=msa,
            substitution=HKY(RealVector(*freqs), Real(kappa)),
            clock=Clock.Strict(Real(clock_rate)),
            tree=tree,
        )

        b3_ll = ll.likelihood()
        newick = tree.to_newick()

        beast1_ll = beast1_likelihood(msa, newick, kappa, freqs, clock_rate)

        diff = abs(b3_ll - beast1_ll)
        rel = diff / abs(beast1_ll)
        assert rel < 1e-10, f"b3={b3_ll}, beast1={beast1_ll}, diff={diff}, rel={rel}"
