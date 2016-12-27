import sys
sys.path.append('../veverica')
tree_adj = {0: {1, 2, 3},
        1: {4, 5},
        2: {6},
        3: {7, 8},
        6: {9, 10},
        }
tree_adj.update({i: set() for i in [4, 5, 9, 10, 7, 8]})
nodes_sign = {4: 1,
        5: -1,
        9: 1,
        10: 1,
        7: -1,
        8: 1,
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
        }
root = 0
correct_answer = {0: (3, 6), 1: (2, 1), 2: (0, 2), 3: (1, 3), 6: (0, 4)}
from shazoo import flep
print(flep(tree_adj, nodes_sign, edge_weight, root))
