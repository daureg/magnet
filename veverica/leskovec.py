import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.utils import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
from collections import defaultdict
from grid_stretch import add_edge, perturbed_bfs
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer
import numpy as np
import real_world as rw
import trolls


def add_neighbor(node, neighbor, G):
    """add `neighbor` to adjacency list of `node`"""
    if node in G:
        G[node].add(neighbor)
    else:
        G[node] = set([neighbor])


def triads_indices(u, v, w, Esign):
    """return triads indices for the triangle (u → v, w)"""
    for e, f in product([(u, w), (w, u)], [(v, w), (w, v)]):
        if e not in Esign or f not in Esign:
            continue
        u_is_endpoint = int(e[0] == u)
        v_is_endpoint = int(e[1] == v)
        se, sf = int(Esign[e]), int(Esign[f])
        yield 8*u_is_endpoint + 4*v_is_endpoint + 2*se + sf


def compute_features(Gin, Gout, G, E, Esign):
    common_nei = {e: G[e[0]].intersection(G[e[1]]) for e in E}    
    knows_indices, pred_indices = [], []    
    features, signs = [], []
    din_plus, dout_plus = defaultdict(int), defaultdict(int)
    din_minus, dout_minus = defaultdict(int), defaultdict(int)
    for (u, v), sign in Esign.items():
        if sign is True:
            din_plus[v] += 1
            dout_plus[u] += 1
        else:
            din_minus[v] += 1
            dout_minus[u] += 1
    for i, ((u, v), sign) in enumerate(E.items()):
        degrees = [len(Gout[u]), len(Gin[v]), len(common_nei[(u, v)]),
                   din_plus[u], din_minus[v], dout_plus[u], dout_minus[v]]
        triads = 16*[0, ]
        for w in common_nei[(u, v)]:
            for t in triads_indices(u, v, w, Esign):
                triads[t] += 1
        features.append(degrees+triads)
        signs.append(sign)
        (knows_indices if (u, v) in Esign else pred_indices).append(i)
    return features, signs, knows_indices, pred_indices



if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    num_threads = 15
    num_rep = 10

    matthews_scorer = make_scorer(matthews_corrcoef)
    rf = RandomForestClassifier(80, n_jobs=num_threads)
    lr = LogisticRegressionCV(Cs=8, n_jobs=num_threads, cv=5,
                              scoring=matthews_scorer, solver='liblinear')

    rw.read_original_graph('soc-sign-Slashdot090221.txt', directed=True)
    G, E = rw.G, rw.EDGE_SIGN
    Gin, Gout = {}, {}
    for u, v in E:
        add_neighbor(u, v, Gout)
        add_neighbor(v, u, Gin)

    rw.DEGREES[-1]
    tree = perturbed_bfs(G, rw.DEGREES[-1][0])
    h, t = zip(*tree)
    lcc = set(h).union(set(t))
    Elcc = {e: s for e, s in E.items() if e[0] in lcc or e[1] in lcc}
    E_nodir =  {((u, v) if u < v else (v, u)): s for (u, v), s in Elcc.items()}
    Glcc = {u: adj for u, adj in G.items() if u in lcc}

    alphas = np.linspace(5, 85, 10)
    res = [[], [], []]
    for alpha in alphas:
        us, rrf, rlr = [], [], []
        alpha /= 100.0
        for _ in range(num_rep):
            alphasign = trolls.select_edges(Glcc, E_nodir, alpha, 'uniform')
            frac = len(alphasign)/len(E_nodir)
            g, pred = trolls.predict_signs(E_nodir, alphasign, .47)
            us.append(trolls.evaluate_pred(g, pred)+[frac])

            directed_edges = {(e if e in Elcc else (e[1], e[0])): s for e, s in alphasign.items()}
            frac = len(directed_edges)/len(Elcc)
            Xall, gold, train, test = compute_features(Gin, Gout, G, Elcc, directed_edges)
            Xa, ya = np.array(Xall), np.array([int(_) for _ in gold])
            rf.fit(Xa[train,:], ya[train])
            pred = rf.predict(Xa[test,:])
            rrf.append([accuracy_score(ya[test], pred), f1_score(ya[test], pred),
                        matthews_corrcoef(ya[test], pred), frac])
            lr.fit(Xa[train,:], ya[train])
            pred = lr.predict(Xa[test,:])
            rlr.append([accuracy_score(ya[test], pred), f1_score(ya[test], pred),
                        matthews_corrcoef(ya[test], pred), frac])
        res[0].append(us)
        res[1].append(rrf)
        res[2].append(rlr)
    import persistent as p
    import time
    p.save_var('{}_{}_{}.my'.format('SLA', int(time.time()), False),
               (alphas, res))
