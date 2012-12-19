"""
:Module: core
:Synopsis: Basic utilities for PST calculus
:Author: dohmatob x dopgima

"""

import re
import numpy as np


def compute_phi(sample, alphabet, node_label):
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
    total = phi.sum()
    if total == 0:
        phi.fill(1.0 / len(alphabet))
    else:
        phi /= total

    return phi


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


def is_parent(x, y):
    return ((get_node_len(x) + 1 == get_node_len(y)) and is_ancestor(x, y))


def get_node_len(node_label):
    return len(get_any_member(node_label))


def are_siblings(x, y):
    return get_suffix(x) == get_suffix(y)


def create_pst(sample, alphabet, order, node_label='', phi=None):
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

    if get_node_len(node_label) <= order:
        if phi is None:
            phi = compute_phi(sample, alphabet, node_label)

        pst[node_label] = phi

        children_pst = {}
        for symbol in alphabet:
            child_node_label = "[%s]%s" % (symbol, node_label)
            child_phi = compute_phi(sample, alphabet, child_node_label)

            # try to lump this child with any siblings that induce thesame
            # conditional distribution
            lump = [
                l for l, p in children_pst.iteritems()
                if are_siblings(l, child_node_label) and np.all(p == phi)]
            if len(lump) == 0:
                # ok, we don't yet have sufficient statistics against this
                # would-be would-be child; so let's add it to the on-going
                # PST anyway
                child_pst = create_pst(sample, alphabet,
                                       order,
                                       phi=child_phi,
                                       node_label=child_node_label,
                                       )
                children_pst.update(child_pst)
            else:
                # it's a bingo! do lumping
                lumped_child_node_label = lump[0]
                assert len(lump) == 1
                del children_pst[lumped_child_node_label]
                lumped_child_node_label = '[' + get_front(
                    lumped_child_node_label)[1:-1] + \
                    symbol + ']' + get_suffix(lumped_child_node_label)
                children_pst[lumped_child_node_label] = child_phi

        pst.update(children_pst)

    return pst


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
                if is_ancestor(anchor_node_label, node_label))


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


def display_pst(pst, level=0, padding=" "):
    if len(pst) > 0:
        for label in pst.keys():
            if get_node_len(label) == level:
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
                        child_pst, level=level + 1, padding=child_padding)
