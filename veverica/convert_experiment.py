#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Convert graph_tool graph to dict of sets and back within the redensify
module."""
import redensify
import random as r
from collections import Counter, deque
from timeit import default_timer
from itertools import repeat, combinations, product, starmap
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
    new_graph()
    redensify.N = graph.num_vertices()
    for node in graph.vertices():
        redensify.G[int(node)] = set(map(int, node.out_neighbours()))
    redensify.EDGES_SIGN.clear()
    has_no_sign = 'sign' not in graph.ep
    for edge in graph.edges():
        src, dst = int(edge.source()), int(edge.target())
        sign = True if has_no_sign else bool(graph.ep['sign'][edge])
        redensify.EDGES_SIGN[(src, dst)] = sign
    finalize_graph()


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
    redensify.PIVOT_SELECTION = kwargs['pivot']
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_rings_experiment(size, nb_rings, shared_sign, rigged, one_at_a_time,
                         pivot=redensify.PivotSelection.Uniform, n_rep=100,
                         pool=None):
    args = repeat({"size": size, "nb_rings": nb_rings, "rigged": rigged,
                   "shared_sign": shared_sign, "pivot": pivot,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_rings, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = [process_rings(_) for _ in args]
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile_name('rings', [size, nb_rings], pivot, one_at_a_time),
               res)


def process_circle(kwargs):
    make_circle(kwargs['circle_size'], kwargs['rigged'])
    redensify.PIVOT_SELECTION = kwargs['pivot']
    return run_one_experiment(100, kwargs['one_at_a_time'])


def run_circle_experiment(size, one_at_a_time, rigged=False, n_rep=100,
                          pivot=redensify.PivotSelection.Uniform, pool=None):
    args = repeat({"circle_size": size, "rigged": rigged, "pivot": pivot,
                   "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_circle, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_circle, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile_name('circle', [size, 0], pivot, one_at_a_time), res)


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


def planted_clusters(ball_size=12, nb_balls=5, pos=False, p=0.07):
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
    flip_random_edges(p)
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
    redensify.PIVOT_SELECTION = kwargs['pivot']
    delta = sum(count_disagreements(true_cluster))
    times, _, errors = run_one_experiment(100, kwargs['one_at_a_time'])
    return [times, delta, errors]


def run_planted_experiment(ball_size, nb_balls, one_at_a_time=True, n_rep=100,
                           pivot=redensify.PivotSelection.Uniform, pool=None):
    args = repeat({"ball_size": ball_size, "nb_balls": nb_balls,
                   "pivot": pivot, "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_planted, args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = list(map(process_planted, args))
    res = {'time': list(map(itemgetter(0), runs)),
           'delta': list(map(itemgetter(1), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile_name('planted', [ball_size, nb_balls], pivot,
                             one_at_a_time), res)


def savefile_name(geometry, params, pivot, one_at_a_time):
    """Create suitable filename to save results"""
    strat = {redensify.PivotSelection.Uniform: 'puni',
             redensify.PivotSelection.ByDegree: 'pdeg',
             redensify.PivotSelection.Preferential:
             'ppre'}[pivot]
    heuristic = 'ONE' if one_at_a_time else 'ALL'
    return '{}_{:04d}_{:03d}_{}_{}_{}.my'.format(geometry, params[0],
                                                 params[1], heuristic, strat,
                                                 int(time.time()))


def random_signed_communities(n_communities, size_communities, degree, p_in,
                              p_pos, p_neg):
    """Create `n_communities` whose number of node is defined by the list
    `size_communities` (or a single integer). Each positive edge within
    communities is created with probability `p_in`. Negative edges are added
    between communities so that the degree of each node is `degree`. Finally
    noise is added. Each positive edge is turned into negative with
    probability `p_neg` and conversely for negative edges and `p_pos`.
    Based on:
    - section 3.3 of Yang, B., Cheung, W., & Liu, J. (2007). Community Mining
    from Signed Social Networks. IEEE Transactions on Knowledge and Data
    Engineering, 19(10), 1333–1348. doi:10.1109/TKDE.2007.1061
    - section IV.A of Jiang, J. Q. (2015). Stochastic Blockmodel and
    Exploratory Analysis in Signed Networks, 12. Physics and Society.
    http://arxiv.org/abs/1501.00594
    """
    if not isinstance(size_communities, list):
        size_communities = n_communities*[size_communities]
    new_graph()

    boundaries = [0]
    clustering = []
    for cluster_id, size in enumerate(size_communities):
        for node in range(size):
            redensify.G[boundaries[-1] + node] = set()
            clustering.append(cluster_id)
        boundaries.append(size + boundaries[-1])

    for nodes in starmap(range, zip(boundaries, boundaries[1:])):
        for i, j in combinations(nodes, 2):
            if r.random() <= p_in:
                add_signed_edge(i, j, True)
    finalize_graph()
    g = to_graph_tool()
    import experiments as xp
    pos = xp.cc.gtdraw.sfdp_layout(g)
    all_nodes = set(range(len(redensify.G)))
    for nodes in starmap(range, zip(boundaries, boundaries[1:])):
        others = list(all_nodes.difference(set(nodes)))
        for u in nodes:
            missing = degree - len(redensify.G[u])
            if missing > 0:
                for v in r.sample(others, missing):
                    add_signed_edge(u, v, False)

    add_noise(p_pos, p_neg)
    finalize_graph()
    # return None, clustering
    return pos.get_2d_array([0, 1]), clustering


def turn_into_signed_graph_at_random(num_cluster=5):
    """Randomly create `num_cluster` clusters in the redensify graph and set
    edges so that there is no disagreements."""
    cluster = [r.randint(0, num_cluster-1) for _ in range(redensify.N)]
    for i, j in redensify.EDGES_SIGN.keys():
        redensify.EDGES_SIGN[(i, j)] = cluster[i] == cluster[j]
    return cluster


def turn_into_2cc_by_breadth_first():
    """Make two balanced clusters by building half of a BF tree from a random
    root"""
    border = deque()
    discovered = {k: False for k in redensify.G}
    # cluster_idx = {k: 0 for k in redensify.G}
    src = r.choice(list(redensify.G.keys()))
    border.append(src)
    discovered[src] = True
    first_cluster_size = 0
    while border and first_cluster_size < 6*len(redensify.G)//10:
        v = border.popleft()
        for w in redensify.G[v]:
            if not discovered[w]:
                discovered[w] = True
                # cluster_idx[w] = 1
                first_cluster_size += 1
                border.append(w)
    cluster_idx = [int(discovered[u]) for u in sorted(redensify.G)]
    for i, j in redensify.EDGES_SIGN.keys():
        redensify.EDGES_SIGN[(i, j)] = cluster_idx[i] == cluster_idx[j]
    return cluster_idx


def turn_into_signed_graph_by_propagation(num_cluster=5,
                                          infected_fraction=0.7):
    """Set the `num_cluster` nodes with highest as cluster centers and
    propagate their label through edges"""
    degrees = {node: len(adj) for node, adj in redensify.G.items()}
    max_degree = max(degrees.values())
    centers = [node for node, deg in degrees.items() if deg == max_degree]
    if len(centers) < num_cluster:
        centers = [_[0] for _ in sorted(degrees.items(),
                                        key=itemgetter(1))[:num_cluster]]
    centers = r.sample(centers, num_cluster)
    unclustered_nodes = set(range(len(degrees)))
    cluster_idx = len(unclustered_nodes)*[0, ]
    for node, degree in degrees.items():
        if degree == 0:
            cluster_idx[node] = r.randint(0, num_cluster-1)
            unclustered_nodes.remove(node)
    for idx, node in enumerate(centers):
        cluster_idx[node] = idx
        unclustered_nodes.remove(node)
    to_cluster = deque(reversed(centers), maxlen=len(degrees))
    while to_cluster:
        u = to_cluster.popleft()
        idx = cluster_idx[u]
        neighbors = redensify.G[u].intersection(unclustered_nodes)
        infected = 0
        for neighbor in neighbors:
            cluster_idx[neighbor] = idx
            unclustered_nodes.remove(neighbor)
            to_cluster.append(neighbor)
            infected += 1
            # quit after propagating to infected_fraction of neighbors
            if infected > infected_fraction*len(neighbors):
                break

    # affect unclustered_nodes at random
    for u in unclustered_nodes:
        cluster_idx[u] = r.randint(0, num_cluster-1)
    for i, j in redensify.EDGES_SIGN.keys():
        redensify.EDGES_SIGN[(i, j)] = cluster_idx[i] == cluster_idx[j]
    return cluster_idx


def add_noise(p_pos, p_neg):
    """Each positive edge is turned into a negative one with probability
    `p_neg` and conversely for negative edges and `p_pos`"""
    for (i, j), sign in redensify.EDGES_SIGN.items():
        if sign:
            if p_neg > 0 and r.random() <= p_neg:
                redensify.EDGES_SIGN[(i, j)] = False
        else:
            if p_pos > 0 and r.random() <= p_pos:
                redensify.EDGES_SIGN[(i, j)] = True


def generate_random_graph(n, pr=0.1):
    """Create an undirected graph of `n` nodes according to Erdős–Rényi model
    with probability of each edge being `pr`."""
    new_graph()
    for node in range(n):
        redensify.G[node] = set()
    for i, j in combinations(range(n), 2):
        if r.random() < pr:
            add_signed_edge(i, j, sign=True)
    finalize_graph()


def fast_random_graph(n, pr=0.1):
    """Create an undirected graph of `n` nodes according to Erdős–Rényi model
    with probability of each edge being `pr`. Use algorithm described in
    Batagelj, Vladimir, and Ulrik Brandes. "Efficient generation of large
    random networks." Physical Review E 71.3 (2005)"""
    import math
    new_graph()
    for node in range(n):
        redensify.G[node] = set()

    v, w = 1, -1
    cst = math.log(1-pr)
    while v < n:
        rnd = r.random()
        w += 1 + int(math.log(1-rnd) / cst)
        while w >= v and v < n:
            w, v = w-v, v+1
        if v < n:
            add_signed_edge(v, w, sign=True)
    finalize_graph()


def preferential_attachment(n, m=1, c=0, gamma=1, bonus_neighbor_prob=0):
    """Create an undirected graph of `n` nodes according to Barabási–Albert
    model where each newly added nodes is connected to `m` previous with
    probability proportional to k**gamma + c (k being degree of the node
    considered).
    http://en.wikipedia.org/wiki/Barabási–Albert_model#Algorithm
    """
    new_graph()
    for node in range(1, m+1):
        add_signed_edge(node-1, node, True)
    degrees = [1, ] + (m-1)*[2, ] + [1, ]
    weights = [_**gamma + c for _ in degrees]
    for new_node in range(m+1, n):
        objects = range(len(degrees))
        num_neighbors = m+1 if r.random() < bonus_neighbor_prob else m
        neighbors = [redensify.weighted_choice(objects, weights)
                     for _ in range(num_neighbors)]
        for n in neighbors:
            add_signed_edge(n, new_node, True)
            degrees[n] += 1
            weights[n] = degrees[n]**gamma + c
        degrees.append(m)
        weights.append(m**gamma + c)
    finalize_graph()


def fast_preferential_attachment(n, m=3, bonus_neighbor_prob=0):
    """Create an undirected graph of `n` nodes according to Barabási–Albert
    model where each newly added nodes is connected to `m` previous with
    probability proportional to k**gamma + c (k being degree of the node
    considered).
    http://en.wikipedia.org/wiki/Barabási–Albert_model#Algorithm
    """
    new_graph()
    for node in range(1, m+1):
        add_signed_edge(node-1, node, True)
    degrees = [1, ] + (m-1)*[2, ] + [1, ]
    candidates = []
    for u, d in enumerate(degrees):
        candidates.extend([u for _ in range(d)])
    for new_node in range(m+1, n):
        num_neighbors = m+1 if r.random() < bonus_neighbor_prob else m
        neighbors = r.sample(candidates, num_neighbors)
        for n in neighbors:
            add_signed_edge(n, new_node, True)
            degrees[n] += 1
            candidates.append(n)
        degrees.append(m)
        candidates.extend([new_node for _ in range(num_neighbors)])
    finalize_graph()
