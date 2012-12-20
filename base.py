import unittest
import re
import numpy as np
from numpy.random import random_sample


def get_node_suffix(x):
    """
    Returns the suffix of a given node x.

    """
    if x == '':
        return x
    else:
        hit = re.match("^\[[^\[\]]+?\](.*)$", x)
        if hit:
            return hit.group(1)
        else:
            hit = re.match("^.(.*)$", x)
            assert not hit is None
            return hit.group(1)


def get_node_prefix(x):
    """
    Returns the prefix of a given node x.

    """
    if x == '':
        return x
    else:
        re.match("^(.*)\[[^\[\]]+?\]$", x).group(1)


def get_any_member(context):
    x = ""
    for lump in re.finditer("\[(.+?)\]", context):
        x = x + lump.group(1)[0]

    return x


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


def is_parent(x, y):
    """
    Determines whether node x is the parent of node y.

    """

    return x != y and get_node_suffix(y) == x


def is_ancestor(x, y):
    """
    Determines whether node x is an ancestor of node y.

    """

    return x != y and y.endswith(x)


def get_node_front(x):
    """
    Returns the front-most symbol in a node label x.

    """

    if x == '':
        return ''
    else:
        return re.match("^(\[[^\[\]]+?\]).*$", x).group(1)


def are_siblings(x, y):
    """
    Determines whether two nodes share a common parent.

    """

    return get_node_suffix(x) == get_node_suffix(y)


def get_node_len(x):
    if x == '':
        return 0
    else:
        return 1 + get_node_len(get_node_suffix(x))


def get_root_node(pst):
    """
    Returns the label of the root node of a PST.

    """

    return pst.keys()[np.argmin([get_node_len(node_label) for node_label
                                in pst.keys()])]


def compute_phi(sample, alphabet, order, node_label, ymin=0.001):
    """
    Given an observed sample and a node label (context), this function
    computes the conditional distribution on the symbols of the
    underlying alphabet.

    Examples
    --------
    >>> from core import *
    >>> compute_phi('aab' * 4, 'ab', 'a')
    array([ 0.5,  0.5])
    >>> compute_phi('aab' * 4, 'ab', 'aa')
    array([ 0.,  1.])
    >>> compute_phi('aab' * 4, 'ab', '')
    array([ 0.66666667,  0.33333333])
    >>>

    """

    phi = np.array(
        [len(re.findall(
                    "%s(?=(%s))" % (node_label, symbol), sample))
         for symbol in alphabet],
        dtype='float')

    # do smoothing
    phi = (1 - order * ymin) * phi + ymin

    total = phi.sum()
    if total == 0:
        phi.fill(1.0 / len(alphabet))
    else:
        phi /= total

    return phi


def change_root(nodes, root, new_root):
    """
    Change the root label of a PST, updating all the other nodes.

    """

    _nodes = {}
    for node_label in nodes:
        _nodes[re.sub(re.escape(root),
                      new_root, node_label)] = nodes[node_label]

    return _nodes


def check_ron_criterion(pst, alphabet, label, phi,
                        pmin=0.001, r=1.05, a=0, ymin=0.001):
    # # if this would-be child node induces thesame (emperical) conditional
    # # distribution as any of its ancestors, then we may ignore it
    # for k, v in pst.iteritems():
    #     if is_ancestor(k, label) and np.all(phi == v):
    #         return False

    # return True

    if label == '':
        return True

    representative = get_any_member(label)
    suffix = get_node_suffix(label)
    if not (get_seq_proba(pst, alphabet, representative) >= pmin):
        return False
    else:
        return np.any((phi >= (1 + a) * ymin) &\
                          ((phi >= r * pst[suffix]) | \
                               (phi <= 1.0 / r * pst[suffix])))


def create_pst(
    sample, alphabet, order, node_label='', phi=None, parent_pst={}):
    """
    Function for creating a PST.

    Parameters
    ----------
    sample: string
        traning data

    alphabet: string or list
        underlying alphabet

    order: int
        maximum order of the VLMC being fitted with the PST.

    node_label: string (optional)
        node_label of the root node of the to-be-constructed PST

    """

    pst = {}

    if phi is None:
        phi = compute_phi(sample, alphabet, order, node_label)

    parent_pst[node_label] = phi

    if get_node_len(node_label) <= order and check_ron_criterion(
        parent_pst, alphabet, node_label, phi):

        pst[node_label] = phi

        children = {}
        for symbol in alphabet:
            child_node_label = "[%s]%s" % (symbol, node_label)
            child_phi = compute_phi(sample, alphabet, order, child_node_label)

            # try to lump this child with any siblings that induce thesame
            # conditional distribution
            lump = [
                l for l, p in children.iteritems()
                if are_siblings(l, child_node_label) and
                np.all(p == child_phi)]
            if len(lump) == 0:
                # ok, we don't yet have sufficient statistics against this
                # would-be would-be child; so let's add it to the on-going
                # PST anyway
                child_pst = create_pst(sample, alphabet,
                                       order,
                                       phi=child_phi,
                                       node_label=child_node_label,
                                       parent_pst=parent_pst
                                       )
                children.update(child_pst)
            else:
                # it's a bingo! do lumping
                lumped_child_node_label = lump[0]
                assert len(lump) == 1
                new_lumped_child_node_label = '[' + get_node_front(
                    lumped_child_node_label)[1:-1] + symbol + ']' + \
                    get_node_suffix(lumped_child_node_label)
                children = change_root(
                    children,
                    lumped_child_node_label, new_lumped_child_node_label)

        pst.update(children)
    else:
        del parent_pst[node_label]

    return pst


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
                next = get_node_suffix(next)
            trans[alphabet[i]] = (correspondence[next], phi[i])
        psa[correspondence[node]] = trans

    return psa


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


def get_children_nodes(pst, node_label):
    """
    Returns all the children node of a given node in a PST.

    Parameters
    ----------
    pst: dict
        handle to the PST under inspection

    node_label: string
        label of the node under inspection


    Examples
    --------

    """

    return dict((other_node_label, pst[other_node_label])
                for other_node_label in pst.keys() if is_parent(
            node_label, other_node_label))


def get_sub_pst(pst, anchor_node_label):
    """
    Returns the sub-PST anchored at a given node.

    Parameters
    ----------
    pst: dict
        handle to the PST under inspection

    anchor_node_label: string
        label of the root node of the sought-for sub-PST

    Examples
    --------

    """

    return dict((node_label, pst[node_label]) for node_label in pst.keys()
                if node_label == anchor_node_label or
                is_ancestor(anchor_node_label, node_label))


def display_pst(pst, padding=" "):
    if not pst:
        return

    label = get_root_node(pst)
    phi = pst[label]

    _padding = padding
    if label == '':
        print "//%s\r\n \\" % str(tuple(phi))
    else:
        print _padding[:-1] + "+-" + label + \
            str(tuple(phi))

    _padding += " "
    children = get_children_nodes(pst, label)

    count = 0
    nchildren = len(children)

    for child_label in children.keys():
        count += 1
        child_padding = _padding
        if count == nchildren:
            child_padding += " "
        else:
            child_padding += "|"

        child_pst = get_sub_pst(pst, child_label)
        display_pst(
            child_pst, padding=child_padding)


class PSTUtilsTest(unittest.TestCase):
    def test_get_node_suffix(self):
        self.assertEqual(get_node_suffix('[a][br]'), '[br]')
        self.assertEqual(get_node_suffix('[t][u]'), '[u]')
        self.assertEqual(get_node_suffix(''), '')

    def test_is_parent(self):
        self.assertTrue(is_parent('[a][bc]', '[cb][a][bc]'))
        self.assertFalse(is_parent('[a][bc]', '[a][bc]'))

    def test_is_ancestor(self):
        self.assertTrue(is_ancestor('[re][ud][de]', '[ty][r][re][ud][de]'))
        self.assertFalse(is_ancestor('[re][ud][de]', '[re][ud][de]'))
        self.assertTrue(is_ancestor('[re][ud][de]', '[ty][re][ud][de]'))

    def test_get_node_front(self):
        self.assertEqual(get_node_front('[a][cr]'), '[a]')
        self.assertEqual(get_node_front('[ty][cry]|o]'), '[ty]')
        self.assertEqual(get_node_front(''), '')

    def test_get_root_node(self):
        pst = {"[a]": [0, 1], "[a][bc]": [0, 0]}
        self.assertEqual(get_root_node(pst), "[a]")

    def test_get_node_len(self):
        self.assertEqual(get_node_len('[ty][re][ud][de]'), 4)
        self.assertEqual(get_node_len('[a][cr]'), 2)
        self.assertEqual(get_node_len('[abcdefghi]'), 1)
        self.assertEqual(get_node_len(''), 0)

    def test_are_siblings(self):
        self.assertTrue(are_siblings('[c][tr]', '[uv][tr]'))
        self.assertTrue(are_siblings('[c][tr]', '[c][tr]'))

    def test_get_children(self):
        pst = {"": [0, 1], "[b]": [0.5, 0.5], "[a]": [0, 1], "[a][bc]": [0, 0]}
        self.assertTrue(get_children_nodes(pst, '[bc]').keys() == ['[a][bc]'])
        self.assertTrue(get_children_nodes(pst, '[b]').keys() == [])
        self.assertTrue(get_children_nodes(pst, '').keys() == ['[a]', '[b]'])

    def test_get_sub_pst(self):
        pst = {"": [0, 1], "[b]": [0.5, 0.5], "[a]": [0, 1], "[a][b]": [0, 0]}
        sub_pst = get_sub_pst(pst, '[b]')
        self.assertTrue(sub_pst.keys() == ['[a][b]', '[b]'])
        sub_pst = get_sub_pst(pst, '[a]')
        self.assertTrue(sub_pst.keys() == ['[a]'])
        self.assertEqual(get_sub_pst(pst, ''), pst)

    def test_create_pst(self):
        pst = create_pst("abracadabra" * 4, 'abcdr', 200)
        print pst_to_psa(pst, 'abcdr')

        display_pst(pst)
        self.assertEqual(
            [x for x in pst.keys() if get_node_suffix(x) in
             pst.keys()], pst.keys())

    def test_change_root(self):
        pst = {"[a]": [0, 1], "[a][bc][a]": [0, 0]}
        self.assertEqual(get_root_node(pst), '[a]')
        pst = change_root(pst, '[a]', '[ab]')
        self.assertEqual(get_root_node(pst), '[ab]')

if __name__ == '__main__':
    unittest.main()
