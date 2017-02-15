#    Copyright 2016 NetworkX developers.
#    Copyright (C) 2004-2016 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.  BSD license.
from future.utils import iteritems
from operator import itemgetter


class UnionFind(object):

    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:
    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.
    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.
      Union-find data structure. Based on Josiah Carlson's code,
      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py
    """

    def __init__(self, elements=None):
        """Create a new empty union-find structure.

        If *elements* is an iterable, this structure will be initialized
        with the discrete partition on the given set of elements.
        """
        if elements is None:
            elements = ()
        self.parents = {}
        self.weights = {}
        for x in elements:
            self.weights[x] = 1
            self.parents[x] = x

    def __getitem__(self, object):
        """Find and return the name of the set containing the object."""
        # check for previously unknown object
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        # Find the heaviest root according to its weight.
        heaviest = max(roots, key=lambda r: self.weights[r])
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def kruskal_mst_edges(edge_weight):
    subtrees = UnionFind()
    res = []
    for (u, v), _ in sorted(iteritems(edge_weight), key=itemgetter(1)):
        if subtrees[u] != subtrees[v]:
            res.append((u, v))
            subtrees.union(u, v)
    return res


def benchmark(ew):
    return sorted(iteritems(ew), key=itemgetter(1))


if __name__ == "__main__":
    from timeit import default_timer as clock
    import sys
    import random
    random.seed(123)
    nrep = 11
    timings = []
    n = int(1e6)
    # ew = {(random.randint(0, n), random.randint(0, n)): 1002*random.random()
    #       for _ in range(30000)}
    # for _ in range(nrep):
    #     start = clock()
    #     benchmark(ew)
    #     timings.append(clock() - start)
    # print('\t'.join(('{:.3g}'.format(t) for t in timings)))
    # print(sum(timings[1:])/(nrep-1))
    # sys.exit()

    import persistent
    dataset = 'usps4500'
    ew, y = persistent.load_var('{}_lcc.my'.format(dataset))
    nrep = 5
    timings = []
    for _ in range(nrep):
        start = clock()
        # mst = [(u, v) for (u, v) in kruskal_mst_edges(ew)]
        mst = kruskal_mst_edges(ew)
        timings.append(clock() - start)
    print('\t'.join(('{:.3g}'.format(t) for t in timings)))
    print(sum(timings[1:])/(nrep-1))
    import networkx as nx
    g = nx.Graph()
    g.add_weighted_edges_from((u,v,w) for (u,v),w in ew.items())
    mst_gold = [(u,v) for (u,v) in nx.minimum_spanning_tree(g).edges()]
    print(sorted(mst) == sorted(mst_gold))
