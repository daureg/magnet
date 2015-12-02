#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Read signed network from SNAP file and try to predict sign on small
subgraph."""
import sys
import os
sys.path.append(os.path.expanduser('~/venvs/34/lib/python3.4/site-packages/'))
import random as r
RND_NBS = [r.random() for _ in range(10000)]
import convert_experiment as cexp
import redensify
G = {}
EDGE_SIGN = {}
DEGREES = None
INCONSISTENT = 0
EDGE_TO_DELETE = {'soc-sign-Slashdot090221.txt': 'universe/slash_delete.my',
                  'soc-sign-epinions.txt': 'universe/epi_delete.my',
                  'soc-mnist.txt': 'universe/mni_delete.my',
                  'soc-mnist_n.txt': 'universe/mnin_delete.my',
                  'soc-wiki.txt': 'universe/wiki_delete.my'}


def add_neighbor(node, neighbor):
    """add `neighbor` to adjacency list of `node`"""
    if node in G:
        G[node].add(neighbor)
    else:
        G[node] = set([neighbor])


def add_signed_edge(a, b, sign, directed=False):
    """Add a `sign`ed edge between `a` and `b`"""
    global INCONSISTENT
    if a > b and not directed:
        a, b = b, a
    e = (a, b)
    if e in EDGE_SIGN:
        if sign != EDGE_SIGN[e]:
            del EDGE_SIGN[e]
            G[a].remove(b)
            G[b].remove(a)
            INCONSISTENT += 1
        return
    add_neighbor(a, b)
    add_neighbor(b, a)
    EDGE_SIGN[e] = sign


def remove_signed_edge(u, v, directed=False):
    if u > v and not directed:
        u, v = v, u
    G[u].remove(v)
    G[v].remove(u)
    del EDGE_SIGN[(u, v)]


def reindex_nodes(old_G, old_E, mapping, directed=False):
    """Change nodes id according to `mapping`"""
    new_G = {}
    for n, adj in old_G.items():
        new_G[mapping[n]] = {mapping[v] for v in adj}
    new_E = {}
    for e, s in old_E.items():
        u, v = mapping[e[0]], mapping[e[1]]
        ne = (u, v) if u < v or directed else (v, u)
        new_E[ne] = s
    return new_G, new_E


def read_original_graph(filename, seed=None, balanced=False, directed=False):
    """Read a signed graph from `filename` and compute its degree sequence.
    Optionally `shuffle` nodes ids"""
    global DEGREES, G, EDGE_SIGN, INCONSISTENT
    DEGREES, G, EDGE_SIGN, INCONSISTENT = None, {}, {}, 0
    with open(filename) as source:
        for line in source:
            if line.startswith('#'):
                continue
            i, j, sign = [int(_) for _ in line.split()]
            if i == j:
                continue
            add_signed_edge(i, j, sign > 0, directed)
    # reindex nodes so they are sequential
    mapping = {v: i for i, v in enumerate(sorted(G.keys()))}
    G, EDGE_SIGN = reindex_nodes(G, EDGE_SIGN, mapping, directed)
    if balanced:
        import persistent
        to_delete = persistent.load_var(EDGE_TO_DELETE[filename])
        for edge in to_delete:
            remove_signed_edge(*edge, directed)
    if isinstance(seed, int):
        r.seed(seed)
        rperm = list(G.keys())
        r.shuffle(rperm)
        rperm = {i: v for i, v in enumerate(rperm)}
        G, EDGE_SIGN = reindex_nodes(G, EDGE_SIGN, rperm, directed)
    DEGREES = sorted(((node, len(adj)) for node, adj in G.items()),
                     key=lambda x: x[1])


def get_ego_nodes(root, hops=3):
    """Get all the nodes connected to `root` by a path of length less than
    `hops`."""
    res = []
    border = [root]
    for step in range(hops):
        new_border = set()
        for n in border:
            res.append(n)
            for neighbor in G[n]:
                if neighbor not in res and neighbor not in border:
                    new_border.add(neighbor)
        border = new_border
    return res


def create_subgraph(nodes):
    """Make a graph from all edges induced by `nodes`."""
    cexp.new_graph()
    for u, node in enumerate(nodes):
        redensify.G[u] = set()
        for neighbor in G[node]:
            if neighbor not in nodes:
                continue
            v = nodes.index(neighbor)
            edge = (node, neighbor) if node < neighbor else (neighbor, node)
            cexp.add_signed_edge(u, v, EDGE_SIGN[edge])
    cexp.finalize_graph()


def find_removable_edges():
    """Return all possible edges that can be removed without creating orphan
    nodes (but the graph might be disconnected)"""
    isolated = [node for node, adj in redensify.G.items() if len(adj) <= 1]
    can_be_removed = []
    degrees = [len(adj) for node, adj in redensify.G.items()]
    for i, j in redensify.EDGES_SIGN.keys():
        if i not in isolated and j not in isolated:
            can_be_removed.append((i, j))
            degrees[i] -= 1
            if degrees[i] <= 1:
                isolated.append(i)
            degrees[j] -= 1
            if degrees[j] <= 1:
                isolated.append(j)
    return can_be_removed


def generate_subgraph_with_test_set(root=100015, random_numbers=None):
    """Extract a subgraph from the big one and remove some edges, that are to
    be predicted."""
    # Slashdot root 37233 (graph size 198?)
    # Epinion root 100015 (graph size 194)
    nodes = get_ego_nodes(root)
    create_subgraph(nodes)
    can_be_removed = find_removable_edges()
    indeed_removed = {}
    edge_idx = 0
    for i, j in can_be_removed:
        edge = (i, j)
        proba = r.random() if not random_numbers else random_numbers[edge_idx]
        edge_idx += 1
        if proba < .3:
            indeed_removed[edge] = redensify.EDGES_SIGN[edge]
            del redensify.EDGES_SIGN[edge]
            redensify.G[i].remove(j)
            redensify.G[j].remove(i)
    cexp.finalize_graph()
    return indeed_removed


def make_prediction(edges_to_be_predicted, cc_run=150):
    """Return the true sign of the `edges_to_be_predicted` along with the
    predicted sign from a clustering."""
    # find a clustering with the fewest disagreements
    best = (1e6, None)
    for i in range(cc_run):
        cluster = cexp.cc_pivot()
        cost = sum(cexp.count_disagreements(cluster))
        if cost < best[0]:
            best = (cost, cluster)

    # Use it to predict the sign of missing edges
    cluster = best[1]
    gold, pred = [], []
    for (i, j), sign in edges_to_be_predicted.items():
        pred.append(cluster[i] == cluster[j])
        gold.append(sign)
    return gold, pred


def binary_score(gold, pred):
    """Evaluate binary classification result `pred` against `gold` values.
    For now this is simply accuracy"""
    return sum((g == p for g, p in zip(gold, pred)))/len(gold)


def process_real(kwargs):
    removed = generate_subgraph_with_test_set(random_numbers=RND_NBS)
    redensify.PIVOT_SELECTION = kwargs['pivot']
    start = cexp.default_timer()
    redensify.complete_graph(one_at_a_time=kwargs['one_at_a_time'])
    elapsed = cexp.default_timer() - start
    gold, pred = make_prediction(removed)
    return elapsed, binary_score(gold, pred)


def run_real(one_at_a_time, pool=None,
             pivot=redensify.PivotSelection.Uniform, n_rep=100):
    args = cexp.repeat({"pivot": pivot, "one_at_a_time": one_at_a_time}, n_rep)

    if pool:
        runs = list(pool.imap_unordered(process_real, args,
                                        chunksize=n_rep//cexp.NUM_THREADS))
    else:
        runs = [process_real(_) for _ in args]
    res = {'time': list(map(cexp.itemgetter(0), runs)),
           'nb_error': list(map(cexp.itemgetter(1), runs))}
    cexp.p.save_var(cexp.savefile_name('real', [0, 0], pivot, one_at_a_time),
                    res)

if __name__ == '__main__':
    from multiprocessing import Pool
    NUM_THREADS = 14
    cexp.NUM_THREADS = NUM_THREADS
    pool = Pool(NUM_THREADS)
    exp_per_thread = 4
    read_original_graph('soc-sign-epinions.txt')
    for i in range(99990,100020):
        print(i, len(get_ego_nodes(i)))
    run_real(one_at_a_time=True, n_rep=exp_per_thread*cexp.NUM_THREADS, pool=None)
    run_real(one_at_a_time=False, n_rep=exp_per_thread*cexp.NUM_THREADS, pool=None)
    run_real(pivot=redensify.PivotSelection.Preferential, one_at_a_time=True,
            n_rep=exp_per_thread*cexp.NUM_THREADS, pool=None)
    run_real(pivot=redensify.PivotSelection.ByDegree, one_at_a_time=False,
            n_rep=exp_per_thread*cexp.NUM_THREADS, pool=None)
    pool.close()
    pool.join()
