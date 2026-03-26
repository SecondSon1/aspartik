from aspartik.b3 import Clock
from aspartik.b3.likelihoods import CPU4Likelihood, Likelihood
from aspartik.b3.parameters import Real, RealVector, Tree
from aspartik.b3.substitutions import HKY, JC, Substiution4
from aspartik.io import read_msa_from_fasta
from aspartik.rng import RNG


class TestAcceptReject:
    def test_reject_restores_exact_value(self, rng: RNG):
        tree, ll = self._load(rng, HKY(RealVector(0.25, 0.25, 0.25, 0.25), Real(2.0)))
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

    def test_multiple_accept_reject_cycles(self, rng: RNG):
        tree, ll = self._load(rng, JC())

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

    def _load(self, rng: RNG, substitution: Substiution4) -> tuple[Tree, Likelihood]:
        msa = read_msa_from_fasta("data/alignments/apes.fasta")
        tree = Tree(msa.sequence_names(), rng)
        ll = CPU4Likelihood(
            msa=msa,
            substitution=substitution,
            clock=Clock.Strict(Real(1.0)),
            tree=tree,
        )
        return tree, ll
