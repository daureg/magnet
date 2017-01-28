import sys
sys.path.append('../veverica')
from shazoo import flep, MAX_WEIGHT, find_hinge_nodes, assign_gamma
from grid_stretch import add_edge, deque, ancestor_info


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
    correct_answer = ((MAX_WEIGHT - 0, MAX_WEIGHT), {}, {0: (True, -1, 0, MAX_WEIGHT, 0, MAX_WEIGHT)})
    assert flep(tree_adj, nodes_sign, edge_weight, root, return_fullcut_info=True) == correct_answer


def test_find_hinge_nodes_paper():
    edge_weight = {(1, 3): 1, (2, 3): 1, (3, 5): 1, (4, 5): 1, (5, 6): 1, (6, 10): 1,
                   (7, 8): 1, (8, 10): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1,
                   (11, 19): 1, (13, 14): 1, (14, 16): 2, (15, 16): 1, (16, 18): 1,
                   (18, 19): 4, (18, 23): 2, (19, 20): 2, (20, 21): 1, (20, 22): 1,
                   (23, 26): 2, (24, 25): 1, (24, 26): 1, (26, 27): 1, (26, 28): 1}
    tree_adj = {}
    for u, v in edge_weight.keys():
        add_edge(tree_adj, u, v)
    nodes_sign = {4: 1, 7: 1, 9: -1, 12: 1, 27: -1, 28: -1}
    correct_answer = {4: 6.75, 7: 5.75, 9: 4.75, 10: 3.75, 11: 2.75, 12: 3.75,
                      26: 2.5, 27: 3.5, 28: 3.5}
    node_to_predict = 14
    answer = find_hinge_nodes(tree_adj, edge_weight, nodes_sign,
                              node_to_predict, with_distances=True)
    assert answer == correct_answer


def test_find_hinge_nodes_random():
    edge_weight = {(0, 1): 1, (0, 5): 2, (0, 6): 1, (0, 7): 1, (0, 8): 2, (0, 9): 4,
                   (0, 13): 1, (0, 20): 2, (0, 21): 1, (0, 23): 2, (0, 27): 2,
                   (1, 2): 2, (1, 4): 5, (1, 16): 3, (1, 25): 3, (2, 3): 4, (2, 10): 2,
                   (2, 22): 2, (4, 11): 2, (6, 12): 4, (6, 17): 3, (8, 19): 1,
                   (9, 14): 2, (9, 15): 5, (9, 18): 1, (9, 24): 2, (9, 29): 2,
                   (20, 26): 3, (20, 28): 1}
    tree_adj = {}
    for u, v in edge_weight.keys():
        add_edge(tree_adj, u, v)
    nodes_sign = { 3: 1, 10: -1, 11: 1, 12: -1, 15: -1, 19: -1, 28: -1, 29: -1}
    correct_answer = {0: 1, 9: 1.25, 1: 2, 2: 2.5,
                      12: .25, 29: 1.75, 15: 1.45, 28: 2.5,
                      19: 2.5, 11: 2.7, 3: 2.75, 10: 3}
    node_to_predict = 6
    answer = find_hinge_nodes(tree_adj, edge_weight, nodes_sign,
                              node_to_predict, with_distances=True)
    assert answer == correct_answer


def test_guilt_twice():
    edge_weight = {(4, 5): 2, (5, 6): 1.5, (6, 10): 2, (7, 8): 3, (8, 10): 1,
                   (9, 10): 2, (10, 11): 1, (11, 12): 2, (11, 19): 1, (18, 19): 4,
                   (18, 23): 2, (23, 26): 2, (26, 27): 1, (26, 28): 1.5}
    tree_adj = {}
    for u, v in edge_weight.keys():
        add_edge(tree_adj, u, v)
    parents = ancestor_info(tree_adj, 11)
    node_signs = {4: -1, 7: 1, 9: 1, 12: -1, 27: 1, 28: -1}
    correct_answer_minus = {4: 1./10, 5: 1./10, 6: 1./10, 7: 0, 8: 1./20, 9: 0, 10: 1./4, 11: 1,
                            12: 1./2, 18: 1./4, 19: 1./4, 23: 1./4, 26: 1./4, 27: 0, 28: 3./20}
    correct_answer_plus = {4: 0, 5: 1./10, 6: 1./10, 7: 1./20, 8: 1./20, 9: 1./10, 10: 1./4, 11: 1,
                           12: 0, 18: 1./4, 19: 1./4, 23: 1./4, 26: 1./4, 27: 1./10, 28: 0}
    minus = assign_gamma(tree_adj, 11, edge_weight, parents, node_signs, -1, only_faulty=False)
    assert correct_answer_minus == minus
    plus = assign_gamma(tree_adj, 11, edge_weight, parents, node_signs, 1, only_faulty=False)
    assert correct_answer_plus == plus
