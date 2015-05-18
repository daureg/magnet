#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Implement the Shazoo node binary classification algorithm from
Vitale, F., Cesa-Bianchi, N., Gentile, C., & Zappella, G. (2011).
See the tree through the lines: the Shazoo algorithm.
In Advances in Neural Information Processing Systems 24 (pp. 1584–1592).
http://eprints.pascal-network.org/archive/00009193/
"""


MAX_WEIGHT = int(2e9)


def flep(tree_adj, nodes_sign, edge_weight, root):
    """Compute the sign of the `root` that yield the smallest weighted cut in
    `tree_adj` given the already revealed `nodes_sign`."""
    assert isinstance(tree_adj, dict)
    stack = []
    status = {_: (False, -1, 0, 0) for _ in tree_adj}
    stack.append(root)
    while stack:
        v = stack.pop()
        if v >= 0:
            discovered, pred, cutp, cutn = status[v]
        else:
            v = -(v+100)
            discovered, pred, cutp, cutn = status[v]
            children = (u for u in tree_adj[v] if status[u][1] == v)
            for child in children:
                edge = (child, v) if child < v else (v, child)
                eweight = edge_weight[edge]
                _, _, childp, childn = status[child]
                cutp += min(childp, childn + eweight)
                cutn += min(childn, childp + eweight)
            status[v] = (discovered, pred, cutp, cutn)
            # print('{}: (+: {}, -: {})'.format(v, cutp, cutn))
            if v == root:
                # return {n: vals[3] - vals[2] for n, vals in status.items()}
                return cutp < cutn

        if not discovered:
            status[v] = (True, pred, cutp, cutn)
            if v in nodes_sign:
                # don't go beyond revealed node
                # TODO don't cross fork either. But how do I know them?
                continue
            stack.append(-(v+100))
            for w in tree_adj[v]:
                discovered, pred, cutp, cutn = status[w]
                if pred == -1:
                    if w in nodes_sign:
                        cutp, cutn = {False: (MAX_WEIGHT, 0),
                                      True: (0, MAX_WEIGHT)}[nodes_sign[w]]
                    status[w] = (discovered, v, cutp, cutn)
                if not discovered:
                    stack.append(w)

if __name__ == '__main__':
    # pylint: disable=C0103
    import convert_experiment as cexp
    import random
    from timeit import default_timer as clock

    def run_once(size):
        cexp.fast_preferential_attachment(size, 1)
        adj = cexp.redensify.G
        ew = {e: 120*random.random() for e in cexp.redensify.EDGES_SIGN}
        ns = {n: random.random() > .5 for n in adj
              if len(adj[n]) == 1 and random.random() < .7}
        root = max(adj.items(), key=lambda x: len(x[1]))[0]
        flep(adj, ns, ew, root)
    run_once(1000)
    run_once(1000)
    start = clock()
    run_once(200000)
    print('done in {:.3f} sec'.format(clock() - start))
