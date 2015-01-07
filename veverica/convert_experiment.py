#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Convert graph_tool graph to dict of sets and back within the redensify
module."""
import redensify
import random as r
from collections import Counter
from timeit import default_timer
from itertools import repeat, combinations, product
from operator import itemgetter
import persistent as p
import time
NUM_THREADS = 0


def make_dash(n, dash_length):
    """put one negative edge every `dash_length` positive ones."""
    start = 0
    res = []
    while start < n:
        src, dst = start, (start+1) % n
        res.append((min(src, dst), max(src, dst)))
        start += 1+dash_length
    return res


def negative_pattern(n, quantity=None, distance=None, dash_length=None):
    """create position for `quantity` negative edges or two of them separated
    by `distance` vertices."""
    assert any([quantity, distance, dash_length]), "give an argument"
    if dash_length and dash_length > 0:
        return make_dash(n, dash_length)
    vertices = list(range(n))
    if quantity:
        starts = sorted(r.sample(vertices, int(quantity)))
        return [(_, (_+1) % n) for _ in starts]
    assert distance <= n//2
    return [(0, 1), (distance, (distance+1) % n)]


def to_python_graph(graph):
    """populate redensify global variables with a representation of `graph`"""
    redensify.N = graph.num_vertices()
    redensify.G.clear()
    redensify.CLOSEABLE_TRIANGLES.clear()
    redensify.TMP_SET.clear()
    for node in graph.vertices():
        redensify.G[int(node)] = set(map(int, node.out_neighbours()))
    redensify.EDGES_SIGN.clear()
    for edge in graph.edges():
        src, dst = (min(map(int, edge)), max(map(int, edge)))
        redensify.EDGES_SIGN[(src, dst)] = bool(graph.ep['sign'][edge])
    redensify.EDGES_ORIG = list(redensify.EDGES_SIGN.keys())


def make_rings(size, nb_rings, shared_sign=True, rigged=False):
    """Create a graph with around `size` nodes splits into `nb_rings`. They
    have one common edge with `shared_sign`. If `rigged` is True, the negative
    edge of each ring is at the middle, or at a position specified by a
    list."""
    if nb_rings == 1:
        return make_circle(size)
    new_graph()
    add_signed_edge(0, 1, shared_sign)
    ring_id = 0

    def add_cycle(length, ring_id):
        start = 1
        # if the shared edge is not positive, then we don't need to add any
        # other negative edges in the rings
        negative_index = r.randint(0, length) if shared_sign else -1
        if negative_index >= 0 and rigged:
            negative_index = int(length/2)
        if negative_index >= 0 and isinstance(rigged, list):
            negative_index = rigged[ring_id] - 1
            assert 0 <= negative_index <= length
        for i in range(length-1):
            end = len(redensify.G)
            add_signed_edge(start, end, i != negative_index)
            start = end
        end = len(redensify.G)
        add_signed_edge(end-1, end, (length - 1) != negative_index)
        add_signed_edge(0, end, length != negative_index)

    for _ in range(nb_rings):
        add_cycle(size//nb_rings, ring_id)
        ring_id += 1
    finalize_graph()


def finalize_graph():
    redensify.EDGES_ORIG = list(redensify.EDGES_SIGN.keys())
    redensify.N = len(redensify.G)
    for v in range(redensify.N):
        redensify.NODE_DEPTH[v] = 1
    for e in redensify.EDGES_ORIG:
        redensify.EDGES_DEPTH[e] = 1


def new_graph():
    redensify.G.clear()
    redensify.CLOSEABLE_TRIANGLES.clear()
    redensify.TMP_SET.clear()
    redensify.EDGES_SIGN.clear()
    redensify.EDGES_DEPTH.clear()
    redensify.NODE_DEPTH.clear()


def make_circle(n, rigged=False):
    new_graph()
    redensify.N = n
    for i in range(n):
        a, p, b = (i-1) % n, i, (i+1) % n
        redensify.G[i] = set([a, b])
        src, dst = min(p, b), max(p, b)
        sign = i != 0 if not rigged else (src, dst) not in rigged
        redensify.EDGES_SIGN[(src, dst)] = sign
    finalize_graph()


def to_graph_tool(run_cc=False):
    import graph_tool as gt
    graph = gt.Graph(directed=False)
    graph.ep['fake'] = graph.new_edge_property('bool')
    graph.ep['sign'] = graph.new_edge_property('bool')
    graph.vp['cluster'] = graph.new_vertex_property('int')
    graph.add_vertex(redensify.N)
    for edge, sign in redensify.EDGES_SIGN.items():
        e = graph.add_edge(edge[0], edge[1])
        graph.ep['sign'][e] = sign
        graph.ep['fake'][e] = edge not in redensify.EDGES_ORIG
    if run_cc:
        all_vert = list(range(redensify.N))
        graph.vp['cluster'].a = itemgetter(*all_vert)(cc_pivot())
    return graph


def cc_pivot():
    """Fill g's cluster_index according to Ailon algorithm"""
    N = redensify.N
    clustered = set()
    unclustered = set(range(N))
    cluster = {}
    current_cluster_index = 0

    def add_to_current_cluster(node):
        cluster[node] = current_cluster_index
        clustered.add(node)
        unclustered.remove(node)

    while unclustered:
        pivot = r.choice(list(unclustered))
        add_to_current_cluster(pivot)
        for n in redensify.G[pivot]:
            edge = (n, pivot) if n < pivot else (pivot, n)
            positive_neighbor = redensify.EDGES_SIGN[edge]
            if positive_neighbor and n in unclustered:
                add_to_current_cluster(n)
        current_cluster_index += 1
    return cluster


def count_disagreements(cluster):
    """Return a boolean edge map of disagreement with current clustering"""

    def disagree(edge, positive):
        return (cluster[edge[0]] == cluster[edge[1]] and not positive) or \
               (cluster[edge[0]] != cluster[edge[1]] and positive)

    return [disagree(edge, redensify.EDGES_SIGN[edge])
            for edge in redensify.EDGES_ORIG]


def process_rings(kwargs):
    make_rings(kwargs['size'], kwargs['nb_rings'], kwargs['shared_sign'],
               kwargs['rigged'])
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_rings_experiment(size, nb_rings, shared_sign, rigged, one_at_a_time,
                         n_rep=100, pool=None):
    args = repeat({"size": size, "nb_rings": nb_rings, "rigged": rigged,
                   "shared_sign": shared_sign,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_rings, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = [process_rings(_) for _ in args]
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile_name('rings', [size, nb_rings], one_at_a_time),
               res)


def process_circle(kwargs):
    make_circle(kwargs['circle_size'], kwargs['rigged'])
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_circle_experiment(size, one_at_a_time, rigged=False, n_rep=100,
                          pool=None):
    args = repeat({"circle_size": size, "rigged": rigged,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_circle, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_circle, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile_name('circle', [size, 0], one_at_a_time), res)


def run_one_experiment(cc_run, one_at_a_time):
    start = default_timer()
    redensify.complete_graph(one_at_a_time=one_at_a_time)
    elapsed = default_timer() - start
    res = []
    for _ in range(cc_run):
        clusters = cc_pivot()
        disagreements = count_disagreements(clusters)
        res.append(sum(disagreements))
    nb_cluster = len(Counter(clusters).keys())
    mean = sum(res)/len(res)
    return elapsed, nb_cluster, mean


def planted_clusters(ball_size=12, nb_balls=5, pos=False):
    new_graph()
    true_cluster = {}
    balls = [make_ball(true_cluster, ball_size) for _ in range(nb_balls-1)]
    filling = int(1.1*ball_size*nb_balls) - len(redensify.G)
    if filling > 0:
        balls.append(make_ball(true_cluster, filling, exact=True))
    if pos:
        from graph_tool import draw as gtdraw
        redensify.EDGES_ORIG = list(redensify.EDGES_SIGN.keys())
        redensify.N = len(redensify.G)
        k = to_graph_tool()
        pos = gtdraw.sfdp_layout(k).get_2d_array([0, 1])
    for b1, b2 in combinations(balls, 2):
        link_balls(b1, b2)
    flip_random_edges()
    finalize_graph()
    return true_cluster, pos


def make_ball(true_cluster, n, exact=False):
    gsize = len(redensify.G)
    if gsize == 0:
        cluster_index = 0
    else:
        cluster_index = true_cluster[gsize-1] + 1
    size = r.randint(int(0.7*n), int(1.3*n))
    if exact:
        size = n
    index = set()
    for _ in range(size):
        v = gsize
        gsize += 1
        redensify.G[v] = set()
        true_cluster[v] = cluster_index
        index.add(v)
    edges = r.sample(list(combinations(index, 2)), int(1.5*size))
    endpoints = set()
    for u, v in edges:
        endpoints.add(u)
        endpoints.add(v)
        add_signed_edge(u, v, True)
    # make sure the ball forms a connected component
    alone = index.difference(endpoints)
    endpoints = list(endpoints)
    for u in alone:
        add_signed_edge(u, r.choice(endpoints), True)
    return index


def add_signed_edge(a, b, sign):
    """add (a,b) with `sign`"""
    a, b = min(a, b), max(a, b)
    if a in redensify.G:
        redensify.G[a].add(b)
    else:
        redensify.G[a] = set([b])
    if b in redensify.G:
        redensify.G[b].add(a)
    else:
        redensify.G[b] = set([a])
    redensify.EDGES_SIGN[(a, b)] = sign


def link_balls(b1, b2):
    edges = r.sample(list(product(b1, b2)), int(1.0*(len(b1)+len(b2))/2))
    for u, v in edges:
        add_signed_edge(u, v, False)


def flip_random_edges(fraction=0.07):
    """Change the sign of `fraction` of `graph` edges"""
    E = list(redensify.EDGES_SIGN.keys())
    for e in r.sample(E, int(fraction*len(E))):
        redensify.EDGES_SIGN[e] = not redensify.EDGES_SIGN[e]


def process_planted(kwargs):
    true_cluster, _ = planted_clusters(kwargs['ball_size'], kwargs['nb_balls'])
    delta = sum(count_disagreements(true_cluster))
    times, _, errors = run_one_experiment(100, kwargs['one_at_a_time'])
    return [times, delta, errors]


def run_planted_experiment(ball_size, nb_balls, one_at_a_time=True, n_rep=100,
                           pool=None):
    args = repeat({"ball_size": ball_size, "nb_balls": nb_balls,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_planted, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_planted, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'delta': list(map(itemgetter(1), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile_name('planted', [ball_size, nb_balls], one_at_a_time),
               res)


def savefile_name(geometry, params, one_at_a_time):
    """Create suitable filename to save results"""
    strat = {redensify.PivotSelection.Uniform: 'puni',
             redensify.PivotSelection.ByDegree: 'pdeg',
             redensify.PivotSelection.Preferential:
             'ppre'}[redensify.PIVOT_SELECTION]
    heuristic = 'ONE' if one_at_a_time else 'ALL'
    return '{}_{:04d}_{:03d}_{}_{}_{}'.format(geometry, params[0], params[1],
                                              heuristic, strat,
                                              int(time.time()))
