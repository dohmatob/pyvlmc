"""
:Module: pst
:Synopsis: Fundamental algorithms in PST calculus.
:Author: dohmatob elvis dopgima
"""

import re
import numpy as np
import unittest
from numpy.random import random_sample


def is_stochastic(phi):
    return np.abs(phi.sum() - 1) < 1e-4


def get_any_member(context):
    x = ""
    for lump in re.finditer("\[(.+?)\]", context):
        x = x + lump.group(1)[0]

    return x


def get_suffix(context):
    x = ''
    hit = re.match("^\[.+?\](.*)$", context)
    if hit:
        x = hit.group(1)
    else:
        hit = re.match("^.(.*)$", context)
        if hit:
            x = hit.group(1)

    return x


def get_front(context):
    x = ''
    hit = re.match("^(\[.+?\]).*$", context)
    if hit:
        x = hit.group(1)
    else:
        hit = re.match("^(.).*$", context)
        if hit:
            x = hit.group(1)

    return x


def is_ancestor(x, y):
    return (y != x and y.endswith(x))


def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]


def simulate_psa(pst, alphabet=None, steps=1000):
    psa = pst_to_psa(pst, alphabet=alphabet)

    state = ''
    count = 0
    print "%35s%35s%35s" % ("current state", "output", "next state")
    while count < steps:
        count += 1
        loco = [(x, y[1]) for x, y in psa[state].iteritems()]
        output = weighted_values(
            np.arange(len(loco)), np.array([z[1] for z in loco]), 1)[0]
        output = loco[output][0]
        next_state = psa[state][output][0]

        print "%35s%35s%35s" % (state, output, next_state)

        state = next_state


def compute_phi(data, order, alphabet, label="", ymin=0.001):
    phi = np.array(
        [len(re.findall(
                    "%s(?=(%s))" % (label, x), data))
         for x in alphabet],
        dtype='float')
    total = phi.sum()
    if total == 0:
        phi.fill(1.0 / len(alphabet))
    else:
        phi /= total

    # do smoothing
    # phi = (1 - order * ymin) * phi + ymin
    assert np.all(phi >= 0)
    phi /= phi.sum()
    assert is_stochastic(phi)

    return phi


def check_ron_criterion(pst, alphabet, label, phi,
                        pmin=0.001, r=1.05, a=0, ymin=0.001):
    # if this would-be child node induces thesame (emperical) conditional
    # distribution as any of its ancestors, then we may ignore it
    for k, v in pst.iteritems():
        if is_ancestor(k, label) and np.all(phi == v):
            return False

    return True

    # representative = get_any_member(label)
    # suffix = get_suffix(label)
    # if not (get_seq_proba(pst, alphabet, representative) >= pmin):
    #     return False
    # else:
    #     return np.any((phi >= (1 + a) * ymin) &\
    #                       ((phi >= r * pst[suffix]) | \
    #                            (phi <= 1.0 / r * pst[suffix])))


def get_context_len(context):
    return len(get_any_member(context))


def are_siblings(x, y):
    return get_suffix(x) == get_suffix(y)


def create_pst(data, order, phi=None, label="",
               parent_pst={}, alphabet=None):
    level = get_context_len(label)
    if level > order:
        return {}

    if alphabet is None:
        alphabet = sorted(list(set(data)))

    if phi is None:
        phi = compute_phi(data, order, alphabet, label=label)

    if level > 0:
        if not check_ron_criterion(parent_pst, alphabet, label, phi):
            return {}

    pst = {label: phi}
    parent_pst.update(pst)
    children = {}

    for symbol in alphabet:
        child_label = "[%s]%s" % (symbol, label)
        # compute (empirical) conditional distribution induced by this
        # context on subsequent symbols
        phi = compute_phi(data, order, alphabet, label=child_label)

        # try to lump this child with any siblings that induce thesame
        # conditional distribution
        lump = [
            l for l, p in children.iteritems()
            if are_siblings(l, child_label) and np.all(p == phi)]
        if len(lump) == 0:
            # ok, we don't yet have sufficient statistics against this would-be
            # would-be child; so let's add it to the on-going PST anyway
            child_pst = create_pst(data, order,
                                   phi=phi,
                                   label=child_label,
                                   parent_pst=parent_pst,
                                   alphabet=alphabet)
            children.update(child_pst)
        else:
            # it's a bingo! do lumping
            lumped_child_label = lump[0]
            assert len(lump) == 1, "%s, %s" % (lump, label)
            del children[lumped_child_label]
            lumped_child_label = '[' + get_front(lumped_child_label)[1:-1] + \
                symbol + ']' + get_suffix(lumped_child_label)
            children[lumped_child_label] = phi

    pst.update(children)
    parent_pst.update(pst)

    return pst


def get_symbol_proba_in_seq(pst, alphabet, s, position):
        """
        This method computes the probability of a symbol at a
        given position in a sequence, w.r.t. our PST.

        """

        assert position < len(s)

        suffix = ""

        ok = False
        for start in xrange(position):
            if ok:
                break
            for suffix in pst.keys():
                if re.match("^%s$" % suffix, s[start:position]):
                    ok = True
                    break

        return pst[suffix][alphabet.index(s[position])]


def get_seq_proba(pst, alphabet, s):
        """
        This method computes the probability of a sequence, w.r.t. our PST.

        """

        return np.prod(
            [get_symbol_proba_in_seq(pst, alphabet, s, position)
             for position in xrange(len(s))])


def pst_to_psa(pst, alphabet=None):
    correspondence = dict((get_any_member(k), k) for k in pst.keys())

    nodes = dict((get_any_member(k), pst[k]) for k in pst.keys())

    if alphabet is None:
        alphabet = sorted(list(set(list("".join(nodes.keys())))))

    psa = {}
    for node, phi in nodes.iteritems():
        trans = dict()
        for i in np.nonzero(phi > 0)[0]:
            next = node + alphabet[i]
            while not next in nodes.keys():
                next = get_suffix(next)
            trans[alphabet[i]] = (correspondence[next], phi[i])
        psa[correspondence[node]] = trans

    return psa


def display_pst(pst, level=0, padding=" "):
    if len(pst) > 0:
        for label in pst.keys():
            if len(get_any_member(label)) == level:
                phi = pst[label]

                _padding = padding
                if label == '':
                    print "//%s\r\n \\" % str(tuple(phi))
                else:
                    print _padding[:-1] + "+-" + label + \
                        str(tuple(phi))

                _padding += " "
                children = dict(
                    (other_label, pst[other_label])
                    for other_label in pst.keys() if re.match(
                        "^\[[^\[\]]+?\]" + re.escape(label) + "$",
                        other_label))

                count = 0
                nchildren = len(children)

                for child_label in children.keys():
                    count += 1
                    child_padding = _padding
                    if count == nchildren:
                        child_padding += " "
                    else:
                        child_padding += "|"
                    child_pst = dict(
                        (other_label, pst[other_label])
                        for other_label in pst.keys()
                        if re.match(
                            ".*" + re.escape(child_label) + "$",
                            other_label))

                    display_pst(
                        child_pst, level=level + 1, padding=child_padding)


class PST(object):
    """
    Encapsulation of Probabilistic/Prediction Suffix Tries.

    """

    splitter = "\0"

    def __init__(self, seq, depth, label="", phi=None, parent=None):
        self.seq = seq
        self.depth = depth
        self.order = depth
        self.pst = {}
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

        self.pst[self.root_path] = self.phi

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

    # def callback(self):
    #     self.summary = self.root_path

    # def child_callback(self, child):
    #     self.summary = child.summary + self.splitter + self.summary

    # def traverse(self, padding=" ", node_callback=None,
    #              child_node_callback=None,
    #              callback_env=None):
    #     if not node_callback is None:
    #         node_callback(self, callback_env)
    #     else:
    #         self.callback()

    #     if self.is_root():
    #         print "//%s\r\n \\" % str(tuple(self.phi))
    #     else:
    #         print padding[:-1] + "+-" + self.root_path + \
    #             str(tuple(self.phi))

    #     padding += " "
    #     nchildren = 0
    #     for _, child in self.children.iteritems():
    #         nchildren += 1
    #         if nchildren == self.nchildren():
    #             child_padding = padding + " "
    #         else:
    #             child_padding = padding + "|"
    #         child.traverse(
    #             padding=child_padding, node_callback=node_callback,
    #             child_node_callback=child_node_callback,
    #             callback_env=callback_env)

    #         if child_node_callback:
    #             child_node_callback(child, callback_env)
    #         else:
    #             self.child_callback(child)

    def check_ron_criterion(self, root_path, phi, pmin, r, a, ymin):
        # # if this would-be child node induces thesame (emperical) conditional
        # # distribution as any of its ancestors, then we may ignore it
        # for k, v in self.pst.iteritems():
        #     if is_ancestor(k, root_path) and np.all(phi == v):
        #         return False

        # return True

        # representative = get_any_member(root_path)
        # suffix = get_suffix(root_path)
        # if not (self.get_seq_proba(representative) >= pmin):
        #     return False
        # else:
        #     return np.any((phi >= (1 + a) * ymin) &\
        #                       ((phi >= r * self.pst[suffix]) |\
        #                            (phi <= 1.0 / r * \
        #                                 self.pst[suffix])))
        return True

    def prune(self, pmin=0.001, r=1.05, a=0, ymin=0.001):
        children = self.children.copy()
        for phi, child in children.iteritems():
            if self.check_ron_criterion(child.root_path, phi,
                                        pmin, r, a, ymin):
                child.pst = self.pst
                child.prune(pmin=pmin, r=r, a=a, ymin=ymin)
            else:
                self.rm_child(child)

    def rm_child(self, child):
        del self.children[tuple(child.phi)]

        for lump in self.pst.keys():
            if lump.startswith(child.root_path):
                del self.pst[lump]

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
            self.pst.update(child.pst)
        else:
            # it's a bingo! do lumping
            assert len(lump) == 1
            child = self.children[tuple(phi)]
            del self.pst[child.root_path]
            if not child.label.startswith('['):
                child.set_label(child.label + label)
            else:
                child.set_label(child.label[1:-1] + label)
            self.pst[child.root_path] = phi

        self.nnodes += child.nnodes

    def compute_phi(self, root_path="", ymin=0.001):
        phi = np.array(
            [len(re.findall(
                        "%s(?=(%s))" % (root_path, x), self.seq))
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
            self.pst = parent.pst

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
            for suffix in self.pst.keys():
                if re.match("^%s$" % suffix, seq[start:position]):
                    ok = True
                    break

        return self.pst[suffix][
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
        training_seq = 'aabaabaabaab' * 100 + 'bceaab' +  'aaabc'* 100 + 'cbabac'
        pst = PST(training_seq, 300)

        # self.assertEqual(pst.label, "")
        # self.assertTrue(pst.is_root())
        # self.assertEqual(pst.order, 3)
        # self.assertEqual(pst.alphabet, ['a', 'b', 'c', 'd', 'r'])
        # self.assertTrue(is_stochastic(pst.phi))

        z = 'abaab'
        display_pst(pst)
        p = pst.get_seq_proba(z)
        print
        print "P(%s) = %s" % (z, pst.get_seq_proba(z))
        print
        print "Prunning .."
        pst.prune()
        print "+++++++Done."
        print
        display_pst(pst)
        print
        print "P(%s) = %s" % (z, pst.get_seq_proba(z))
        q = pst.get_seq_proba(z)
        self.assertEqual(p, q)
        print
        print pst_to_psa(pst)
        simulate_psa(pst)

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
