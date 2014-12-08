#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Precompute various property of all possible edges triplet"""
import persistent as p
from enum import Enum, unique
from itertools import product
# pylint: disable=C0103


@unique
class TriangleStatus(Enum):
    """What kind of triangle are we picking as candidates for potential
    closing"""
    closeable = 1
    one_edge_missing = 2
    one_edge_positive = 3
    any = 4
    closed = 5

edge_types = [None, False, True]
tr_type = list(product(edge_types, repeat=3))
edge_score = lambda e: -10 if e is None else (e*2 - 1)
# A triangle is closeable if one edge is missing and at least another one is
# positive
triangle_is_closeable = {tr: sum((edge_score(_) for _ in tr)) in [-10, -8]
                         for tr in tr_type}
# Tell if a triangle has 3 edges
triangle_is_closed = {tr: sum((edge_score(_) for _ in tr)) >= -3
                      for tr in tr_type}
any_ = {_: True for _ in tr_type}
one_missing = {_: sum((e is None for e in _)) == 1 for _ in tr_type}
one_positive = {_: sum((e is True for e in _)) >= 1 for _ in tr_type}

TriangleStatusCache = {TriangleStatus.closeable.value: triangle_is_closeable,
                       TriangleStatus.one_edge_missing.value: one_missing,
                       TriangleStatus.one_edge_positive.value: one_positive,
                       TriangleStatus.any.value: any_,
                       TriangleStatus.closed.value: triangle_is_closed}

if __name__ == '__main__':
    p.save_var('triangle_cache.my', TriangleStatusCache)
