import logging
import os
import random
from glob import glob
from itertools import product
from math import sqrt
from multiprocessing import Pool
from timeit import default_timer as clock

import msgpack

import convert_experiment as cexp
import pack_graph as pg
from create_dssn_pack import irregularities
from grid_stretch import make_grid, perturbed_bfs
from new_galaxy import galaxy_maker
from random_tree import get_tree as get_rst
from scratch import average_strech, merge_into_2_clusters


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


def doubling_size(topo):
    size = 1024
    while size < 25e6:
        get_graph(topo, False, '', size)
        size *= 2


def distribute_tasks(tasks, num_threads):
    tasks.sort(key=lambda x: (x[0], x[1][2]))
    threads_tasks = [list() for i in range(num_threads)]
    for i, (_, task) in enumerate(tasks):
        threads_tasks[num_threads - i % num_threads - 1].append(tuple(task))
    for thread_tasks in threads_tasks:
        random.shuffle(thread_tasks)
    return [task for thread_tasks in threads_tasks for task in thread_tasks]


def build_trees(topo, real, num_rep, part):
    prefix = 'nantes/{}_{}*.pack'.format(topo, 'yes' if real else 'no')

    def is_packed_tree(name):
        return 'gtx' in name or 'bfs' in name or 'rst' in name
    graphs = sorted(((f, os.stat(f).st_size) for f in glob(prefix) if not is_packed_tree(f)),
                    key=lambda x: x[1])
    tasks = []
    for filename, size_b in graphs:
        prefix = os.path.splitext(filename)[0]
        for i, treekind in product(range(num_rep), ['bfs', 'gtx', 'rst']):
            tid = part * num_rep + i
            tree_filename = '{}_{}_{}.pack'.format(prefix, treekind, tid)
            if not os.path.exists(tree_filename):
                tasks.append((size_b, (filename, treekind, tid)))
    num_threads = 8
    tasks = distribute_tasks(tasks, num_threads)
    pool = Pool(num_threads)
    pool.starmap_async(single_tree, tasks, chunksize=max(1, len(tasks) // num_threads))
    pool.close()
    pool.join()


def single_tree(filename, treekind, tid):
    prefix = os.path.splitext(filename)[0]
    tree_filename = '{}_{}_{}.pack'.format(prefix, treekind, tid)
    if os.path.exists(tree_filename):
        return
    logging.info('building {} {} on {}'.format(treekind, tid, prefix))
    _, _, _, G, E = load_graph(filename)
    if treekind == 'bfs':
        root = max(G.items(), key=lambda x: len(x[1]))[0]
        t = perturbed_bfs(G, root)
    elif treekind == 'gtx':
        t, _ = galaxy_maker(G, 150, short=True, output_name=None)
    elif treekind == 'rst':
        t = list(get_rst(G, {e: 1 for e in E})[1])
    logging.info('computing stretch of {} on {}'.format(treekind, prefix))
    stretch = average_strech(set(E), t)
    with open(tree_filename, 'w+b') as outfile:
        msgpack.pack((stretch, tuple((x for u, v in t for x in (u, v)))), outfile)


if __name__ == "__main__":
    import socket
    import argparse
    part = int(socket.gethostname()[-1]) - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("topology", help="Which data to use", default='grid',
                        choices={'PA', 'grid', 'triangle'})
    parser.add_argument("-r", "--real", action='store_true', help="Use real data")
    parser.add_argument("-n", "--nrep", help="Number of repetitions", type=int, default=4)
    args = parser.parse_args()
    # doubling_size(args.topology)
    dbg_fmt = '%(asctime)s - %(relativeCreated)d:%(filename)s.%(funcName)s:%(lineno)d: %(message)s'
    logging.basicConfig(filename='trees_{}.log'.format(args.topology),
                        level=logging.DEBUG, format=dbg_fmt)
    build_trees(args.topology, args.real, args.nrep, part)
