from collections import Counter
from math import sqrt
from timeit import default_timer as clock

import msgpack

import convert_experiment as cexp
import pack_graph as pg
from create_dssn_pack import irregularities
from grid_stretch import make_grid
from scratch import merge_into_2_clusters


def load_graph(filename):
    with open(filename, 'r+b') as outfile:
        psi, rV, phi, rE = msgpack.unpack(outfile, use_list=False)
    G, E = {}, {}
    for u, v, s in zip(rE[::3], rE[1::3], rE[2::3]):
        E[(u, v)] = bool(s)
        pg.add_edge(G, u, v)
    nodes_sign = list(rV)
    return psi, phi, nodes_sign, G, E


def to_python_graph(graph):
    G = {int(u): {int(v) for v in u.out_neighbours()} for u in graph.vertices()}
    E = {(int(e.source()), int(e.target())): True for e in graph.edges()}
    return G, E


def get_graph(topology, real, name, size=None):
    if real and topology == 'PA':
        size = {'aut': 4773, 'wik': 7065, 'sla': 82052,
                'epi': 119070, 'kiw': 137713, 'gplus': 74917}[name]
    filename = 'nantes/{}_{}_{}_{}.pack'.format(topology, 'yes' if real else 'no', name, size)
    try:
        return load_graph(filename)
    except FileNotFoundError:
        start = clock()
        print('creating {}'.format(filename))
    if real:
        assert topology == 'grid' and name in {'nips_logo', 'nips_poster', 'space',
                                               'waterfall', 'zmonastery', 'zworld'}
        import convert_pbm_images as pbm
        nodes_sign, G, E = pbm.build_graph(*pbm.read_img('nantes/{}_{}.pbm'.format(name, size)))
        psi, phi = irregularities(G, nodes_sign, E)
        nodes_sign = [nodes_sign[i] for i in range(len(G))]
    else:
        if topology == 'PA':
            cexp.fast_preferential_attachment(size, 3, .13)
        if topology == 'grid':
            G, E_keys = make_grid(int(sqrt(size)))
            cexp.redensify.G = G
            cexp.redensify.N = len(G)
            cexp.redensify.EDGES_SIGN = {e: True for e in E_keys}
        if topology == 'triangle':
            import graph_tool.generation as gen
            import numpy as np
            points = np.random.random((size, 2)) * (size // 50 + 1)
            g, _ = gen.triangulation(points, type="delaunay")
            cexp.redensify.G, cexp.redensify.EDGES_SIGN = to_python_graph(g)
            cexp.redensify.N = size
        n = cexp.redensify.N
        nb_cluster = int(2 * sqrt(n))
        ci = cexp.turn_into_signed_graph_by_propagation(nb_cluster,
                                                        infected_fraction=.9)
        G, E = dict(cexp.redensify.G), dict(cexp.redensify.EDGES_SIGN)
        _, nodes_sign = merge_into_2_clusters(E, ci)
        psi, phi = irregularities(G, dict(enumerate(nodes_sign)), E)
    with open(filename, 'w+b') as outfile:
        msgpack.pack((psi, tuple(map(int, nodes_sign)), phi,
                      tuple((x for (u, v), s in E.items() for x in (u, v, int(s))))), outfile)
    print('save a {} nodes, {} edges in {:.3f} seconds'.format(len(G), len(E), clock() - start))

if __name__ == "__main__":
    import sys
    topo = sys.argv[1]
    assert topo in {'PA', 'grid', 'triangle'}
    size = 1024
    while size < 25e6:
        get_graph(topo, False, '', size)
        size *= 2
