#! /usr/bin/env python
# vim: set fileencoding=utf-8
from copy import deepcopy
import random
from math import log2, ceil
import itertools


class Tree(object):
    """A simple binary tree where each node keep the sum of its subtree and
    each leaves refer to a indices"""
    __slots__ = 'val left right wleft wright'.split()

    def __init__(self, val, left, right, wleft, wright):
        self.val = val
        self.left = left
        self.right = right
        self.wleft = wleft
        self.wright = wright


class WeightedDegrees(object):
    """Dynamically keep degrees of node in memory and sample them
    proportionally to them (potentially after applying some function)"""

    def __init__(self, degrees, func, with_replacement=False):
        self.degrees = deepcopy(degrees)
        self.map = func or (lambda x: x)
        self.build_tree()

    def build_tree(self):
        self.weights = [self.map(_) for _ in self.degrees]
        self.num_active = sum((1 for _ in self.degrees if _ > 0))
        # TODO get rid of 0 padding by setting not existing right nodes on the
        # border to None
        tree_height = ceil(log2(len(self.weights)))
        padding = (2**tree_height - len(self.weights)) * [0, ]
        self.weights += padding
        leaves = [Tree(i, None, None, v if i % 2 == 0 else 0,
                       v if i % 2 == 1 else 0)
                  for i, v in enumerate(self.weights)]
        nodes = leaves
        while len(nodes) > 1:
            pairs = 2*[iter(nodes)]
            next_level = []
            for u, v in itertools.zip_longest(*pairs):
                next_level.append(Tree(None, u, v, u.wleft+u.wright,
                                       v.wleft+v.wright))
            nodes = next_level
        self.root = nodes[0]
        self.path = {i: self.path_from_number(i, tree_height)
                     for i in range(len(self.weights))}

    @staticmethod
    def path_from_number(x, tree_height):
        string = bin(x)[2:]
        string = (tree_height - len(string))*'0' + string
        return [c == '1' for c in string]

    def sample(self, target=None, with_replacement=False):
        total_weight = self.root.wleft + self.root.wright
        target = target or random.random()*total_weight
        current = self.root
        path = []
        prev = 0
        while current.val is None:
            l, r = current.wleft, current.wright
            left_interval = (prev, prev+l)
            right_interval = (prev+l, prev+l+r)
            if left_interval[0] <= target < left_interval[1]:
                path.append(False)
                current = current.left
            else:
                assert right_interval[0] <= target < right_interval[1]
                path.append(True)
                current = current.right
                prev += l
        result = current.val
        if not with_replacement:
            # set weight to 0 so that result will never be sampled again
            self.update_weights({result: -self.degrees[result]})
        return current.val

    def update_weights(self, updates):
        for node, delta in updates.items():
            self.degrees[node] += delta
            if self.degrees[node] < 1e-7:
                new_weight = 0
                self.num_active -= 1
            else:
                new_weight = self.map(self.degrees[node])
            self.update_weight_along_path(node,
                                          new_weight - self.weights[node])
            self.weights[node] = new_weight

    def update_weight_along_path(self, elem, update):
        current = self.root
        for go_right in self.path[elem]+[elem % 2 == 1]:
            if go_right:
                current.wright += update
                current = current.right
            else:
                current.wleft += update
                current = current.left

    def pop(self):
        return self.sample(with_replacement=False)

    def __len__(self):
        return self.num_active


if __name__ == '__main__':
    # pylint: disable=C0103
    weights = [5, 50, 10, 4, 1]
    res = [i for i, v in enumerate(weights) for _ in range(v)]

    sampler = WeightedDegrees(weights, lambda x: x)
    all([sampler.sample(i, with_replacement=True) == gold
         for i, gold in enumerate(res)])

    import numpy as np
    from math import exp, log
    n = 18
    weights = np.random.rand(n)*5
    # weights /= weights.sum()
    weights[n//2] = 0
    weighting = lambda x: exp(-2*x)+log(x+1)
    sampler = WeightedDegrees(weights, weighting)
    delta0, weights[4] = 2.3 - weights[4], 2.3
    delta1, weights[5] = 3.2 - weights[5], 3.2
    sampler.update_weights({4: delta0, 5: delta1})
    samples = np.array([sampler.sample(with_replacement=True)
                        for _ in range(30000)])
    count = np.bincount(samples, minlength=n).astype(float)
    count /= count.sum()
    weights = np.vectorize(weighting)(weights)
    weights /= weights.sum()
    print(np.allclose(weights, count, rtol=1e-2, atol=1e-2))
    print(weights[4], count[4])
    print(weights[5], count[5])
    print(weights)
    print(count)
