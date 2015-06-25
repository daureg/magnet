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
        self.high_nodes = set(self.high.keys())
        self.adjacency = adjacency
        self.used = set()

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
            # remove zero degree nodes, more precisely those who became part
            # of a star during the last sampling step
            if queue[u] == delta and u in self.used:
                self.num_active -= 1
                del queue[u]
                if queue is self.high:
                    self.high_nodes.remove(u)
            else:
                queue[u] -= delta

        if self.num_active == 0:
            return
        # make sure that all degrees in high are higher than those in low
        low_max = -1 if not self.low else -self.low.peekitem()[1]
        while self.high and self.high.peekitem()[1] < low_max:
            node, degree = self.high.popitem()
            self.high_nodes.remove(node)
            self.low[node] = -degree

        new_size_of_high = self.threshold_func()
        while self.high and len(self.high) > new_size_of_high:
            # high is too big so move small degree to low
            node, degree = self.high.popitem()
            self.high_nodes.remove(node)
            self.low[node] = -degree
        while self.low and len(self.high) < new_size_of_high:
            # high is too small so move low big degree to high
            node, degree = self.low.popitem()
            self.high_nodes.add(node)
            self.high[node] = -degree
        # print('{} nodes, threshold at {}: {}/{}'.format(self.num_active,
        #                                                 new_size_of_high,
        #                                                 len(self.high),
        #                                                 len(self.low)))

    def sample_node(self):
        deltas = defaultdict(int)
        high_size, low_size = len(self.high), len(self.low)
        assert self.num_active == high_size + low_size
        assert high_size == self.threshold_func()
        chosen = random.choice(list(self.high_nodes))
        deltas[chosen] = self.high[chosen]
        self.used.add(chosen)
        points = []
        for v in self.adjacency[chosen]:
            if v in self.used:
                continue
            self.used.add(v)
            if v in self.high_nodes:
                deltas[v] = self.high[v]
            else:
                deltas[v] = -self.low[v]
            points.append(v)
            for w in self.adjacency[v]:
                if w in self.used or w == chosen:
                    continue
                deltas[w] = 1
        self.update_degrees(deltas)
        return chosen, points

    def __len__(self):
        return self.num_active

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
