"""
:Module: pst
:Synopsis: Fundamental algorithms in PST calculus.
:Author: dohmatob elvis dopgima
"""

import re
import numpy as np
import unittest


def is_stochastic(phi):
    return np.abs(phi.sum() - 1) < 1e-4


class PST(object):
    """
    Encapsulation of Probabilistic/Prediction Suffix Tries.

    """

    splitter = "\0"

    def __init__(self, seq, depth, label="", phi=None, parent=None):
        self.seq = seq
        self.depth = depth
        self.order = depth
        self.lumped_nodes = {}
        self.set_parent(parent)
        self.set_label(label)
        self.compute_alphabet()
        self.nnodes = 1
        self.children = {}
        self.summary = ""

        if phi is None:
            self.phi = self.compute_phi()
        else:
            self.phi = phi

        self.lumped_nodes[self.root_path] = self.phi

        self._grow()

    def set_label(self, label):
        if len(label) < 1:
            self.label = label
            self.root_path = label
        else:
            self.label = "[%s]" % label
            self.root_path = self.label + self.parent.root_path

    def _grow(self):
        if self.depth > 0:
            for label in self.alphabet:
                self._bear_child(label)

    def callback(self):
        self.summary = self.root_path

    def child_callback(self, child):
        self.summary = child.summary + self.splitter + self.summary

    def traverse(self, padding=" ", node_callback=None,
                 child_node_callback=None,
                 callback_env=None):
        if not node_callback is None:
            node_callback(self, callback_env)
        else:
            self.callback()

        if self.is_root():
            print "//%s\r\n \\" % str(tuple(self.phi))
        else:
            print padding[:-1] + "+-" + self.root_path + \
                str(tuple(self.phi))

        padding += " "
        nchildren = 0
        for _, child in self.children.iteritems():
            nchildren += 1
            if nchildren == self.nchildren():
                child_padding = padding + " "
            else:
                child_padding = padding + "|"
            child.traverse(
                padding=child_padding, node_callback=node_callback,
                child_node_callback=child_node_callback,
                callback_env=callback_env)

            if child_node_callback:
                child_node_callback(child, callback_env)
            else:
                self.child_callback(child)

    def get_any_member(self, pattern):
        x = ""
        for lump in re.finditer("\[(.+?)\]", pattern):
            x = x + lump.group(1)[0]

        return x

    def get_suffix(self, pattern):
        x = ''
        hit = re.match("^\[.+?\](.*)$", pattern)
        if hit:
            x = hit.group(1)

        return x

    def check_ron_criterion(self, root_path, phi, pmin, r, a, ymin):
        representative = self.get_any_member(root_path)
        suffix = self.get_suffix(root_path)
        if not (self.get_seq_proba(representative) >= pmin):
            return False
        else:
            return np.any((phi >= (1 + a) * ymin) &\
                              ((phi >= r * self.lumped_nodes[suffix]) |\
                                   (phi <= 1.0 / r * \
                                        self.lumped_nodes[suffix])))

    def prune(self, pmin=0.001, r=1.05, a=0, ymin=0.001):
        children = self.children.copy()
        for phi, child in children.iteritems():
            if self.check_ron_criterion(child.root_path, phi,
                                        pmin, r, a, ymin):
                child.lumped_nodes = self.lumped_nodes
                child.prune(pmin=pmin, r=r, a=a, ymin=ymin)
            else:
                self.rm_child(child)

    def rm_child(self, child):
        del self.children[tuple(child.phi)]

        for lump in self.lumped_nodes.keys():
            if lump.startswith(child.root_path):
                del self.lumped_nodes[lump]

        print "Removed %s and all it descendants." % child.root_path

    def _bear_child(self, label, pmin=0.001, r=1.05, a=0, ymin=0.001):
        # compute root path of would-be new child node
        child_root_path = "[%s]%s" % (label, self.root_path)

        # compute (empiral) conditional distribution induced by this
        # context on subsequent symbols
        phi = self.compute_phi(root_path=child_root_path, ymin=ymin)

        # try to lump this child with order any siblings that induce thesame
        # conditional distribution
        lump = [
            c.label for phi_tilde, c in self.children.items()
            if np.all(phi_tilde == phi)]
        if len(lump) == 0:
            # Oops! no luck! now check Ron's criterion
            if not self.check_ron_criterion(child_root_path, phi,
                                            pmin, r, a, ymin):
                # it'd be a waste of resources to add this node, ignore it
                return

            # ok, we don't yet have sufficient statistics to justify the
            # 'impertinence' of this node, so add it to the PST anyway
            child = PST(
                self.seq, self.depth - 1, label=label,
                phi=phi, parent=self)
            self.children[tuple(phi)] = child
            self.lumped_nodes.update(child.lumped_nodes)
        else:
            # it's a bingo! do lumping
            assert len(lump) == 1
            child = self.children[tuple(phi)]
            del self.lumped_nodes[child.root_path]
            if not child.label.startswith('['):
                child.set_label(child.label + label)
            else:
                child.set_label(child.label[1:-1] + label)
            self.lumped_nodes[child.root_path] = phi

        self.nnodes += child.nnodes

    def compute_phi(self, root_path="", ymin=0.001):
        phi = np.array(
            [len(re.findall(
                        "%s%s" % (root_path, x), self.seq))
             for x in self.alphabet],
            dtype='float')
        total = phi.sum()
        if total == 0:
            phi.fill(1.0 / len(self.alphabet))
        else:
            phi /= total

        # do smoothing
        phi = (1 - self.order * ymin) * phi + ymin
        phi /= phi.sum()
        assert is_stochastic(phi)
        return phi

    def set_parent(self, parent):
        self.parent = parent

        if parent:
            self.order = parent.order
            self.lumped_nodes = parent.lumped_nodes

    def compute_alphabet(self):
        self.alphabet = list(set(self.seq))
        self.alphabet.sort()

    def is_root(self):
        return not self.parent

    def is_leaf(self):
        return self.nchildren() == 0

    def nchildren(self):
        return len(self.children)

    def get_symbol_proba_in_seq(self, seq, position):
        """
        This method computes the probability of a symbol at a
        given position in a sequence, w.r.t. our PST.

        """

        assert position < len(seq)

        suffix = ""

        ok = False
        for start in xrange(position):
            if ok:
                break
            for suffix in self.lumped_nodes.keys():
                if re.match("^%s$" % suffix, seq[start:position]):
                    ok = True
                    break

        return self.lumped_nodes[suffix][
            self.alphabet.index(seq[position])]

    def get_seq_proba(self, seq):
        """
        This method computes the probability of a sequence, w.r.t. our PST.

        """

        return np.prod(
            [self.get_symbol_proba_in_seq(seq, position)
             for position in xrange(len(seq))])


class PSTTest(unittest.TestCase):
    def test__init(self):
        training_seq = 'abracadabra' * 10 + "abr" + 'abracadadabra' * 15
        pst = PST(training_seq, 5)

        # self.assertEqual(pst.label, "")
        # self.assertTrue(pst.is_root())
        # self.assertEqual(pst.order, 3)
        # self.assertEqual(pst.alphabet, ['a', 'b', 'c', 'd', 'r'])
        # self.assertTrue(is_stochastic(pst.phi))

        z = 'abracb'
        pst.traverse()
        p = pst.get_seq_proba(z)
        print
        print "P(%s) = %s" % (z, pst.get_seq_proba(z))
        print
        print "Prunning .."
        pst.prune()
        print "+++++++Done."
        print
        pst.traverse()
        print
        print "P(%s) = %s" % (z, pst.get_seq_proba(z))
        q = pst.get_seq_proba(z)
        self.assertEqual(p, q)

if __name__ == '__main__':
    unittest.main()

    # seq = 'abracadabra' * 2500
    # pst = PST(seq, 5)
    # pst.do_leafs()
    # print pst.get_seq_proba('abrac')
    # # import pylab as pl
    # # counts = [PST('aabc' * 10, j).counts() for j in xrange(7)]

    # # _, ax = pl.subplots()
    # # pl.plot(counts)
    # # pl.legend(("With lumping", "Without lumping"))
    # # pl.xlabel("Order (L) of PST")
    # # pl.ylabel("Number of nodes in PST")
    # # ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # # pl.show()
