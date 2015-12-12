import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.utils import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)
from collections import defaultdict
from grid_stretch import add_edge, perturbed_bfs
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, make_scorer
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
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


def compute_more_features(din, dout, common_nei, G, E, Esign, with_triads=True):
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
        known_out = dout_plus[u]+dout_minus[u]
        known_in = din_plus[v]+din_minus[v]
        # degrees = [len(Gout[u]), len(Gin[v]), len(common_nei[(u, v)]),
        #            din_plus[u], din_minus[v], dout_plus[u], dout_minus[v]]
        degrees = [dout[u], din[v], len(common_nei[(u, v)]),
                   din_plus[u], din_minus[v], dout_plus[u], dout_minus[v],
                   din[u], dout[v],
                   din_plus[v], din_minus[u], dout_plus[v], dout_minus[u],
                   dout_minus[u]/dout[u], din_minus[v]/din[v],
                   0 if known_out == 0 else dout_minus[u]/known_out,
                   0 if known_in == 0 else din_minus[v]/known_in,
                   ]
        triads = 16*[0, ]
        if with_triads:
            for w in common_nei[(u, v)]:
                for t in triads_indices(u, v, w, Esign):
                    triads[t] += 1
        features.append(degrees+triads)
        signs.append(int(sign))
        (knows_indices if (u, v) in Esign else pred_indices).append(i)
    return features, signs, knows_indices, pred_indices

def us_predict(features):
    return np.logical_or(features[:, 0] < .305, features[:, 1] < .13)

def us_predict2(features):
    return np.logical_or(features[:, 1] < .387, features[:, 0] < .02)

def us_predict3(features):
    return features[:, 1] < (.13 + (features[:, 0]<.3)*.54)

def us_predictlr(features, frac=1):
    coeff = (1-(1-frac)*(-.1))
    return (-6.89*features[:, 0]-6.761*features[:, 1]+5.334) > 0.5/coeff


def pred_simplest(features):
    return features[:, 0] < .5


def pred_fixed(features):
    return features[:, 1] < (.5 + (features[:, 0] < .5)*1)


def pred_tuned(features):
    return np.logical_or(features[:, 0] < .315, features[:, 1] < .096)


def pred_tuned_cmx(features):
    return features[:, 1] < (.096 + (features[:, 0]<.316)*(.759-.096))


def append_res(prev_res, s, end, pred, gold, frac):
    fp = confusion_matrix(gold, pred)[0, 1]/len(pred)
    prev_res.append([accuracy_score(gold, pred), f1_score(gold, pred),
                     matthews_corrcoef(gold, pred), fp, end-s, frac])

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    from copy import deepcopy
    import time
    import sys
    import persistent as p
    import socket
    from timeit import default_timer as clock
    part = int(socket.gethostname()[-1])-1
    num_threads = 16
    num_rep = int(sys.argv[2])

    matthews_scorer = make_scorer(matthews_corrcoef)
    # rf = RandomForestClassifier(80, n_jobs=num_threads, max_features=.5,
    #                             criterion='entropy', class_weight='balanced')
    # lr = LogisticRegressionCV(Cs=np.logspace(-3, 4, 10), n_jobs=num_threads, cv=4,
    #                           scoring=matthews_scorer, solver='lbfgs',
    #                           class_weight={0: 1.4, 1: 1})
    olr = LogisticRegression(C=5.455, solver='lbfgs', n_jobs=num_threads,
                             warm_start=True)
    llr = LogisticRegression(C=1e-3, solver='lbfgs', n_jobs=num_threads,
                             warm_start=True)
    dt = DecisionTreeClassifier(criterion='gini', max_features=None,
                                max_depth=2, class_weight={0: 1.4, 1: 1})
    data = {'WIK': 'soc-wiki.txt',
            'EPI': 'soc-sign-epinions.txt',
            'SLA': 'soc-sign-Slashdot090221.txt'}
    pref = sys.argv[1]
    assert pref in data.keys()
    rw.read_original_graph(data[pref], directed=True)
    G, E = rw.G, rw.EDGE_SIGN
    dout, din = defaultdict(int), defaultdict(int)
    for u, v in E:
        dout[u] += 1
        din[v] += 1

    rw.DEGREES[-1]
    tree = perturbed_bfs(G, rw.DEGREES[-1][0])
    h, t = zip(*tree)
    lcc = set(h).union(set(t))
    Elcc = {e: s for e, s in E.items() if e[0] in lcc or e[1] in lcc}
    E_nodir =  {((u, v) if u < v else (v, u)): s for (u, v), s in Elcc.items()}
    Glcc = {u: adj for u, adj in G.items() if u in lcc}
    common_nei = {e: G[e[0]].intersection(G[e[1]]) for e in Elcc}
    Gout, Gin = {}, {}
    for u, v in Elcc:
        add_neighbor(u, v, Gout)
        add_neighbor(v, u, Gin)

    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    feats = list(range(7)) + list(range(17, 33))
    alphas = np.linspace(0, 65, 10)
    res = [[], [], [], [], [], [], []]
    auc = 0
    for alpha in alphas:
        lesko, fixed, simple_fixed, tuned, cmp_tuned, dectree, logreg = [], [], [], [], [], [] ,[]
        alpha /= 100.0
        for _ in range(num_rep):
            Eout = trolls.select_edges(Gout, Elcc, alpha, 'uniform', True)
            Ein = trolls.select_edges(Gin, Elcc, alpha, 'uniform', True)
            directed_edges = deepcopy(Ein)
            directed_edges.update(Eout)
            frac = len(directed_edges)/len(Elcc)
            Xall, gold, train, test = compute_more_features(din, dout,
                                                            common_nei, G, Elcc,
                                                            directed_edges,
                                                            with_triads=True)
            Xa, ya = np.array(Xall), np.array(gold)
            train_feat = np.ix_(train, feats)
            test_feat = np.ix_(test, feats)

            s = clock()
            pred = pred_fixed(Xa[test, 15:17])
            end = clock()
            append_res(fixed, s, end, pred, ya[test], frac)

            s = clock()
            pred = pred_simplest(Xa[test, 15:17])
            end = clock()
            append_res(simple_fixed, s, end, pred, ya[test], frac)

            s = clock()
            pred = pred_tuned(Xa[test, 15:17])
            end = clock()
            append_res(tuned, s, end, pred, ya[test], frac)

            s = clock()
            pred = pred_tuned_cmx(Xa[test, 15:17])
            end = clock()
            append_res(cmp_tuned, s, end, pred, ya[test], frac)

            s = clock()
            llr.fit(Xa[train_feat], ya[train])
            pred = llr.predict(Xa[test_feat])
            end = clock()
            append_res(lesko, s, end, pred, ya[test], frac)

            s = clock()
            olr.fit(Xa[train, 15:17], ya[train])
            pred = olr.predict(Xa[test, 15:17])
            end = clock()
            append_res(logreg, s, end, pred, ya[test], frac)

            s = clock()
            dt.fit(Xa[train, 15:17], ya[train])
            pred = dt.predict(Xa[test, 15:17])
            end = clock()
            append_res(dectree, s, end, pred, ya[test], frac)

        res[0].append(lesko)
        res[1].append(fixed)
        res[2].append(simple_fixed)
        res[3].append(tuned)
        res[4].append(cmp_tuned)
        res[5].append(logreg)
        res[6].append(dectree)

    p.save_var('{}_{}_{}.my'.format(pref, start, part+1), (alphas, res))
