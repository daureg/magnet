#! /usr/bin/env python
# vim: set fileencoding=utf-8
""""""
from collections import defaultdict
import random


class ThresholdSampler(object):
    """given a list of node degree, return star centers, sampling from the ones
    currently below threshold"""

    def __init__(self, adjacency, threshold):
        """Build the initial list"""
        assert set(adjacency.keys()) == set(range(len(adjacency)))
        self.nodes = sorted(((n, len(a)) for n, a in adjacency.items()),
                            key=lambda x: -x[1])
        self.nodes_to_pos = {v[0]: i for i, v in enumerate(self.nodes)}
        self.num_active = len(self.nodes)
        self.adjacency = adjacency
        self.used = [False for u in adjacency]
        self.threshold = threshold

    def update_weight(self, deltas):
        for u, delta in deltas.items():
            index = self.nodes_to_pos[u]
            _, degree = self.nodes[index]
            new_degree = degree + delta
            self.nodes[index] = (u, new_degree)
            if new_degree <= 0:
                self.nodes[index] = (u, 0)
                if degree > 0:
                    self.num_active -= 1
        self.nodes = sorted(self.nodes, key=lambda x: -x[1])
        self.nodes_to_pos = {v[0]: i for i, v in enumerate(self.nodes)}

    def _node_degree(self, node):
        return self.nodes[self.nodes_to_pos[node]][1]

    def sample_node(self):
        deltas = defaultdict(int)
        assert self.num_active == sum((1 for _ in self.nodes if _[1] > 0))
        chosen_index = random.randint(0, int(self.threshold(self.num_active)))
        chosen = self.nodes[chosen_index][0]
        deltas[chosen] = -self._node_degree(chosen)
        self.used[chosen] = True
        for v in self.adjacency[chosen]:
            if self.used[v]:
                continue
            deltas[v] = -self._node_degree(v)
            self.used[v] = True
            for w in self.adjacency[v]:
                if self.used[w]:
                    continue
                deltas[w] -= 1
        self.update_weight(deltas)
        return chosen

if __name__ == '__main__':
    # pylint: disable=C0103
    import redensify
    from convert_experiment import fast_preferential_attachment
    fast_preferential_attachment(200, 3, .13)
    from math import log
    from timeit import default_timer as clock
    for _ in range(5):
        start = clock()
        centers = ThresholdSampler(redensify.G, log)
        while centers.num_active > 0:
            (centers.sample_node())
        print('{:.3f} seconds'.format(clock() - start))
    # print(centers.nodes)
    # print(centers.nodes_to_pos)
    # print(centers.sample_node())
    # print(centers.nodes)
    # print({i: v for i, v in enumerate(centers.used)})
    # print(centers.num_active)
