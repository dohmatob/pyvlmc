"""
:Module: pst
:Synopsis: Fundamental algorithms in PST calculus.
:Author: dohmatob elvis dopgima
"""

import re
import numpy as np


class PST(object):
    """
    Encapsulation of Probabilistic/Prediction Suffix Tries.

    """

    def __init__(self, seq, L, label="", parent=None, padding=" "):
        self.label = label
        self.nodes = {}
        self.root_path = ""
        self.children = {}
        self.set_parent(parent)

        if not self.is_root():
            self.root_path = str(label) + self.root_path

        self.seq = seq
        self.alphabet = "".join(set(seq))
        self.grow(seq, L, padding)

    def add_child(self, child):
        self.children[child.label] = child

    def is_root(self):
        return not self.parent

    def set_parent(self, parent):
        self.parent = parent

        if parent:
            self.root_path = parent.root_path
            parent.add_child(self)

    def grow(self, seq, L, padding):
        """
        Given a string/sequence seq, and an order L, this method grow PST
        down a depth L.

        """

        d = len(self.alphabet)

        p = np.array(
            [len(re.findall("%s%s" % (self.root_path, a), "".join(seq)))
             for a in self.alphabet], dtype='float64')

        r = p.sum()
        if r == 0:
            p.fill(1.0 / d)
        else:
            p /= r

        if self.is_root():
            print "//%s\r\n \\" % str(tuple(p))
        else:
            print padding[:-1] + "+-" + self.root_path + str(tuple(p)) + "/"
        padding += " "

        self.cool = self.root_path
        self.nodes = {self.root_path: p}

        if L > 0:
            for j in xrange(d):
                a = self.alphabet[d - j - 1]
                 # expand child
                child_padding = padding
                if j + 1 == d:
                    child_padding += ' '
                else:
                    child_padding += '|'

                child = PST(seq, L - 1, label=a, parent=self,
                            padding=child_padding)
                self.cool = self.cool + "0" + child.cool
                self.nodes.update(child.nodes)

    def get_symbol_proba_in_seq(self, seq, position):
        """
        This method computes the probability of a symbol at a
        given position in a sequence, w.r.t. our PST.

        """

        if position == 0:
            suffix = self.root_path
        else:
            for start in xrange(position):
                hit = re.search(seq[start:position], self.cool)
                if not hit is None:
                    suffix = hit.group()
                    break

        return self.nodes[suffix][
            self.alphabet.index(seq[position])]

    def get_seq_proba(self, seq):
        """
        This method computes the probability of a sequence, w.r.t. our PST.

        """

        return np.prod(
            [self.get_symbol_proba_in_seq(seq, position)
             for position in xrange(len(seq))])

#######
# DEMO
#######
if __name__ == '__main__':
    pst = PST('aabaabaabaab', 12)
    z = 'abaab'
    print "P(%s) = %f" % (z, pst.get_seq_proba(z))
