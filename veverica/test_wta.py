# vim: set fileencoding=utf-8
from __future__ import division

import random

import pytest

from grid_stretch import add_edge
from wta import (LineNode, convert_to_line, dfs_order, linearize_tree,
                 predict_signs, propagate_revelead_signs, remove_duplicates)


def test_propagate_revelead_signs():
    n = 9
    nodes = [LineNode(id_=i) for i in range(n)]
    for i, ln in enumerate(nodes):
        if i > 0:
            ln.left = nodes[i-1]
        if i < n-1:
            ln.right = nodes[i+1]
    nodes[2].sign = 1
    nodes[6].sign = -1
    weights = [1, 1/2, 1/2, 3, 1, 1/2, 1/2, 1/2]
    ew = {(u, v): w for u, v, w in zip(range(n), range(1, n), weights)}
    propagate_revelead_signs(nodes, ew)
    answer = [(u.sign if u.sign is not None else u.pred_sign, u.source, u.smallest_dst)
              for u in nodes]
    correct_anwser = [(1, 2, 3),
                      (1, 2, 2),
                      (1, 2, 0),
                      (1, 2, 2),
                      (1, 2, 2+1/3),
                      (-1, 6, 2),
                      (-1, 6, 0),
                      (-1, 6, 2),
                      (-1, 6, 4),
                      ]
    assert answer == correct_anwser
    answer = propagate_revelead_signs(nodes, ew, return_prediction=True)
    correct_anwser = {0: 1, 1: 1, 3: 1, 4: 1, 5: -1, 7: -1, 8: -1}
    assert answer == correct_anwser


@pytest.fixture
def tree():
    ew = {(1, 5): 4,
          (1, 6): 1,
          (1, 8): 1/2,
          (0, 6): 3,
          (3, 6): 3,
          (6, 7): 1/2,
          (2, 8): 3,
          (4, 8): 1/2}
    tree_adj = {}
    for u, v in ew.keys():
        add_edge(tree_adj, u, v)
    return tree_adj, ew


def test_dfs_order(tree):
    tree_adj, ew = tree
    correct_anwser = [1, 5, 1, 6, 0, 6, 3, 6, 7, 6, 1, 8, 2, 8, 4]
    answer = dfs_order(tree_adj, ew, 1)
    assert answer == correct_anwser


def test_dfs_order_big_leaf():
    ew = {(1, 2): 4,
          (1, 3): 1,
          (1, 100): 1/2,
          (2, 4): 3,
          (2, 5): 3,
          (3, 6): 1/2}
    tree_adj = {}
    for u, v in ew.keys():
        add_edge(tree_adj, u, v)
    correct_anwser = [1, 2, 4, 2, 5, 2, 1, 3, 6, 3, 1, 100]
    answer = dfs_order(tree_adj, ew, 1)
    assert answer == correct_anwser


def test_remove_duplicates(tree):
    tree_adj, ew = tree
    initial_order = [1, 5, 1, 6, 0, 6, 3, 6, 7, 6, 1, 8, 2, 8, 4]
    correct_anwser = [(1, 4), (5, 1), (6, 3), (0, 3), (3, 1/2),
                      (7, 1/2), (8, 3), (2, 1/2), (4, None)]
    answer = remove_duplicates(initial_order, ew)
    assert answer == correct_anwser


def test_convert_line():
    node_and_weight = [(1, 4), (5, 1), (6, 3), (0, 3), (3, 1/2),
                       (7, 1/2), (8, 3), (2, 1/2), (4, None)]
    nodes = {u: LineNode(id_=u) for u, _ in node_and_weight}
    nodes[1].right = nodes[5]
    nodes[5].right = nodes[6]
    nodes[6].right = nodes[0]
    nodes[0].right = nodes[3]
    nodes[3].right = nodes[7]
    nodes[7].right = nodes[8]
    nodes[8].right = nodes[2]
    nodes[2].right = nodes[4]
    nodes[5].left = nodes[1]
    nodes[6].left = nodes[5]
    nodes[0].left = nodes[6]
    nodes[3].left = nodes[0]
    nodes[7].left = nodes[3]
    nodes[8].left = nodes[7]
    nodes[2].left = nodes[8]
    nodes[4].left = nodes[2]
    correct_anwser = ([nodes[u] for u, _ in node_and_weight],
                      {(1, 5): 4, (5, 6): 1, (0, 6): 3, (0, 3): 3,
                       (3, 7): 1/2, (7, 8): 1/2, (2, 8): 3, (2, 4): 1/2})
    answer = convert_to_line(node_and_weight)
    assert answer == correct_anwser


def test_predict_signs(tree):
    tree_adj, ew = tree
    root = 1
    signs = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: -1, 6: -1, 7: -1, 8: -1}
    n = len(signs)
    nodes_line, edge_weight = linearize_tree(tree_adj, ew, root)
    for i in range(10):
        training_set = {u: signs[u] for u in [random.randint(0, n-1), random.randint(0, n-1)]}
        predict_signs(nodes_line, edge_weight, training_set)
    training_set = {u: signs[u] for u in [2, 6]}
    answer = predict_signs(nodes_line, edge_weight, training_set)
    correct_anwser = {0: -1, 1: -1, 3: -1, 4: 1, 5: -1, 7: 1, 8: 1}
    assert answer == correct_anwser
