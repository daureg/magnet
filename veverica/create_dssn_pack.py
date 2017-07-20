from collections import Counter

import msgpack

import pack_graph as pg


def add_neighbor(G, node, neighbor):
    """add `neighbor` to adjacency list of `node`"""
    if node in G:
        G[node].add(neighbor)
    else:
        G[node] = set([neighbor])


def add_signed_edge(G, E, a, b, sign, directed=False):
    """Add a `sign`ed edge between `a` and `b`"""
    global INCONSISTENT
    if a > b and not directed:
        a, b = b, a
    e = (a, b)
    if e in E:
        if sign != E[e]:
            del E[e]
            G[a].remove(b)
            G[b].remove(a)
            INCONSISTENT += 1
        return
    add_neighbor(G, a, b)
    add_neighbor(G, b, a)
    E[e] = sign


def reindex_nodes(old_G, old_E, mapping, directed=False):
    """Change nodes id according to `mapping`"""
    new_G = {}
    for n, adj in old_G.items():
        if n not in mapping:
            continue
        new_G[mapping[n]] = {mapping[v] for v in adj if v in mapping}
    new_E = {}
    for e, s in old_E.items():
        if e[0] not in mapping or e[1] not in mapping:
            continue
        u, v = mapping[e[0]], mapping[e[1]]
        ne = (u, v) if u < v or directed else (v, u)
        new_E[ne] = s
    return new_G, new_E


def generate_node_sign(G, E):
    import numpy as np
    import scipy.sparse as sparse
    from scipy.optimize import minimize
    def fun(x):
        t = -As@x
        return x.T@t, 2 * t
    rows, cols, data = [], [], []
    for (i, j), s in E.items():
        rows.append(i)
        cols.append(j)
        data.append(2 * s - 1)
        rows.append(j)
        cols.append(i)
        data.append(2 * s - 1)
    As = sparse.csc_matrix(sparse.coo_matrix((data, (rows, cols)), shape=(len(G), len(G))))
    x1 = 2 * np.random.rand(len(G)) - 1
    x1b = 2 * (x1 > 0).astype(int) - 1
    print('Initialy: {} disagrements'.format((fun(x1b)[0] // 2 + len(E)) // 2))
    res = minimize(fun, x1, jac=True, bounds=[(-1, 1) for _ in x1],
                   options={'maxiter': 150000, 'maxfun': 150000,})
    xfb = 2 * (res.x > 0).astype(int) - 1
    print(str(res.message), res.success, res.nfev, (fun(xfb)[0] // 2 + len(E)) // 2)
    return xfb


def irregularities(G, V, E):
    phi_edges, psi = 0, 0
    for (u, v), positive in E.items():
        phi_edges += int((V[u] == V[v] and not positive) or (V[u] != V[v] and positive))
    for u, adj in G.items():
        plus_minus_count = Counter(E[(u, v) if u < v else (v, u)] for v in adj).values()
        if len(plus_minus_count) == 2:
            psi += min(plus_minus_count)
    return psi // 2, phi_edges


if __name__ == "__main__":
    import graph_tool as gt
    from graph_tool.topology import label_largest_component
    import numpy as np
    for name in ('aut', 'wik', 'sla', 'epi', 'kiw'):
        dG, dE = pg.load_directed_signed_graph('directed_{}.pack'.format(name))
        print(name)
        uG, uE = {}, {}
        INCONSISTENT = 0
        for (u, v), s in dE.items():
            add_signed_edge(uG, uE, u, v, s)
        print(len(dG), len(uG))
        print(len(dE), len(uE), INCONSISTENT)
        g = gt.Graph(directed=False)
        g.add_edge_list(uE.keys())
        lcc = label_largest_component(g)
        print(lcc.a.sum())
        G, E = reindex_nodes(uG, uE, {v: i for i, v in enumerate(np.where(lcc.a)[0])})
        print(len(G), len(E), Counter(E.values()))
        node_signs = generate_node_sign(G, E)
        psi, phi = irregularities(G, dict(enumerate(node_signs)), E)
        psi, phi
        filename = 'nantes/PA_yes_{}_{}.pack'.format(name, len(G))
        with open(filename, 'w+b') as outfile:
            msgpack.pack((psi, tuple(map(int, node_signs)), phi,
                          tuple((x for (u, v), s in E.items() for x in (u, v, int(s))))), outfile)
