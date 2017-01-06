import sys
sys.path.append('../veverica')
from shazoo import flep, MAX_WEIGHT


def test_flep_larger_tree():
    tree_adj = {0: {1, 2, 3},
                1: {4, 5},
                2: {6},
                3: {7, 8, 11},
                6: {9, 10},
                11: {12, 13}
                }
    tree_adj.update({i: set() for i in [4, 5, 9, 10, 7, 8, 12, 13]})
    nodes_sign = {4: 1,
                  5: -1,
                  9: 1,
                  10: 1,
                  7: -1,
                  8: 1,
                  12: -1,
                  13: -1,
                  }
    edge_weight = {(0,1): 4,
                   (0, 2): 6,
                   (0, 3): 3,
                   (1, 4): 1,
                   (1, 5): 2,
                   (2, 6): 2,
                   (3, 7): 1,
                   (3, 8): 3,
                   (6, 9): 2,
                   (6, 10): 2,
                   (3, 11): 1,
                   (11, 12): 2,
                   (11, 13): 3,
                   }
    root = 0
    correct_answer = {0: (4, 6), 1: (2, 1), 2: (0, 2), 3: (1, 3), 6: (0, 4),
                      11: (5, 0), 3: (2, 3)}
    assert flep(tree_adj, nodes_sign, edge_weight, root, return_fullcut_info=True)[1] == correct_answer


def test_flep_one_level_tree():
    tree_adj = {0: {1, 2, 3, 4},
                }
    tree_adj.update({i: set() for i in [1, 2, 3, 4]})
    nodes_sign = {1: 1,
                  2: 1,
                  3: 1,
                  4: -1,
                  }
    edge_weight = {(0,1): 1,
                   (0, 2): 1,
                   (0, 3): 0.9,
                   (0, 4): 3,
                   }
    root = 0
    correct_answer = {0: (3, 2.9)}
    assert flep(tree_adj, nodes_sign, edge_weight, root, return_fullcut_info=True)[1] == correct_answer


def test_flep_line():
    tree_adj = {0: {1},
                1: {2},
                2: {3},
                3: set(),
                }
    nodes_sign = {3: 1}
    edge_weight = {(0,1): 1,
                   (1, 2): 3,
                   (2, 3): 1,
                   }
    root = 0
    correct_answer = {0: (0, 1),
                      1: (0, 1),
                      2: (0, 1),
                      }
    assert flep(tree_adj, nodes_sign, edge_weight, root, return_fullcut_info=True)[1] == correct_answer


def test_flep_leaf():
    tree_adj = {0: set()}
    nodes_sign = {0: 1}
    edge_weight = {}
    root = 0
    correct_answer = (MAX_WEIGHT - 0, {})
    assert flep(tree_adj, nodes_sign, edge_weight, root, return_fullcut_info=True) == correct_answer
