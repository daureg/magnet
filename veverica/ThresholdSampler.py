#! /usr/bin/env python
# vim: set fileencoding=utf-8
""""""
from collections import defaultdict
from heap.heap import heap
import random


class ThresholdSampler(object):
    """given a list of node degree, return star centers, sampling from the ones
    currently below threshold"""

    def __init__(self, adjacency, threshold_func):
        """Build the two initial priority queues"""
        assert set(adjacency.keys()) == set(range(len(adjacency)))
        degrees = sorted(((u, len(adj)) for u, adj in adjacency.items()),
                         key=lambda x: -x[1])
        self.num_active = len(degrees)
        self._threshold_func = lambda x: int(threshold_func(x))
        high_size = self.threshold_func()
        self.high = heap({u: degree
                          for i, (u, degree) in enumerate(degrees)
                          if i < high_size})
        self.low = heap({u: -degree
                         for i, (u, degree) in enumerate(degrees)
                         if i >= high_size})
        self.adjacency = adjacency
        self.used = [False for u in adjacency]

    def threshold_func(self):
        non_empty = min(self.num_active, 1)
        return max(non_empty, self._threshold_func(self.num_active))

    def update_degrees(self, deltas):
        # perform actual update
        for u, delta in deltas.items():
            if u in self.high:
                queue = self.high
            else:
                queue = self.low
                delta = -delta
            queue[u] -= delta
            # remove zero degree nodes
            if abs(queue[u]) < 1:
                self.num_active -= 1
                del queue[u]

        if self.num_active == 0:
            return
        # make sure that all degrees in high are higher than those in low
        low_max = -1 if not self.low else -self.low.peekitem()[1]
        while self.high and self.high.peekitem()[1] < low_max:
            node, degree = self.high.popitem()
            self.low[node] = -degree

        new_size_of_high = self.threshold_func()
        while self.high and len(self.high) > new_size_of_high:
            # high is too big so move small degree to low
            node, degree = self.high.popitem()
            self.low[node] = -degree
        while self.low and len(self.high) < new_size_of_high:
            # high is too small so move low big degree to high
            node, degree = self.low.popitem()
            self.high[node] = -degree
        # print('{} nodes, threshold at {}: {}/{}'.format(self.num_active,
        #                                                 new_size_of_high,
        #                                                 len(self.high),
        #                                                 len(self.low)))

    def _node_degree(self, node):
        if node in self.high:
            return self.high[node]
        return -self.low[node]

    def sample_node(self):
        deltas = defaultdict(int)
        high_size, low_size = len(self.high), len(self.low)
        assert self.num_active == high_size + low_size
        assert high_size == self.threshold_func()
        chosen = random.choice(list(self.high.keys()))
        deltas[chosen] = self._node_degree(chosen)
        self.used[chosen] = True
        for v in self.adjacency[chosen]:
            if self.used[v]:
                continue
            self.used[v] = True
            deltas[v] = self._node_degree(v)
            for w in self.adjacency[v]:
                if self.used[w]:
                    continue
                deltas[w] = 1
        self.update_degrees(deltas)
        return chosen

if __name__ == '__main__':
    # pylint: disable=C0103
    import redensify
    from convert_experiment import fast_preferential_attachment
    fast_preferential_attachment(2*5000, 3, .13)
    from math import log
    from timeit import default_timer as clock
    for _ in range(5):
        start = clock()
        centers = ThresholdSampler(redensify.G, log)
        while centers.num_active > 0:
            (centers.sample_node())
        print('{:.3f} seconds'.format(clock() - start))
