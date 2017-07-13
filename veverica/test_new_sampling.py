import random

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

import LillePrediction as llp

graph = llp.LillePrediction(use_triads=False)
graph.load_data(llp.lp.DATASETS.Epinion)
graph.select_train_set(sampling=lambda d: int(.02*d))
graph.compute_features()
idx2edge = {i: e for e, i in graph.edge_order.items()}
olr = LogisticRegression(C=.02, warm_start=True, solver='lbfgs', n_jobs=16)

def perf(gold, pred):
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    return ([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
            matthews_corrcoef(gold, pred), fp/(fp+tn)])


def run_once(graph, olr):
    Xl, yl, train_set, test_set = graph.compute_features()
    Xa, ya = np.array(Xl), np.array(yl)
    gold = ya[test_set]
    olr.fit(Xa[train_set, 15:17], ya[train_set])
    pred = olr.predict(Xa[test_set, 15:17])

    pred_reciprocal = []
    changes = 0
    for e, p in zip(test_set, pred):
        er = graph.reciprocal.get(e)
        if er is None or idx2edge[er] not in graph.Esign:
            pred_reciprocal.append(p)
        else:
            twin_sign = graph.Esign[idx2edge[er]]
            changes += int(twin_sign != p)
            pred_reciprocal.append(twin_sign)
    return perf(gold, pred) + perf(gold, pred_reciprocal)


def sophisticated_select_edge(graph, alpha):
    Gout, Gin, E = graph.Gout, graph.Gin, graph.E
    res = {}
    sampling = lambda d: int(alpha*d)
    for u, adj in Gout.items():
        num_to_sample = max(min(len(adj), 1), sampling(len(adj)))
        nei = []
        for v in adj:
            feats = []
            e = (u, v)
            twin = graph.reciprocal.get(e)
            if twin is None:
                feats.append(0)
            else:
                feats.append(1 if twin in res else -1)
            feats.append(len(Gin[v]))
            # add randomness between otherwise equal edges
            feats.append(random.randint(0, 10000))
            feats.append(e)
            nei.append(feats)
        nei.sort()
        res.update({_[-1]: E[_[-1]] for _ in nei[:num_to_sample]})
    for u, adj in Gin.items():
        num_to_sample = max(min(len(adj), 1), sampling(len(adj)))
        nei = []
        for v in adj:
            feats = []
            e = (v, u)
            twin = graph.reciprocal.get(e)
            if twin is None:
                feats.append(0)
            else:
                feats.append(1 if twin in res else -1)
            feats.append(len(Gout[v]))
            feats.append(random.randint(0, 10000))
            feats.append(e)
            nei.append(feats)
        nei.sort()
        res.update({_[-1]: E[_[-1]] for _ in nei[:num_to_sample]})
    return res

nrep = 8
old_alpha = np.zeros((nrep, 8))
new_alpha = np.zeros((nrep, 8))
np.set_printoptions(linewidth=120)
# sla .28 .3065
# wik .295 .3057
# epi .261 .292
for i in range(nrep):
    Esign=graph.select_train_set(sampling=lambda d: int(.261*d))
    old_alpha[i, :] = run_once(graph, olr)
    print(old_alpha[i, :])
    newE = sophisticated_select_edge(graph, .292)
    graph.Esign = newE
    new_alpha[i, :] = run_once(graph, olr)
    print(new_alpha[i, :])
    np.savez_compressed('new_samp_res_epi', before=old_alpha, after=new_alpha)
