from timeit import default_timer as clock

import numpy as np

from next_state_distrib import next_state_distrib


def compute_bayes_features(Xa, ya, train_set, test_set, graph):
    prior_n, prior_p = np.bincount(ya[train_set])/len(train_set)
    epsilon = 1e-10

    def node_quantity(args):
        u, kinp, kinn, koutp, koutn = args
        dinp, dinn = graph.din_plus[u]+kinp, graph.din_minus[u]+kinn,
        doutp, doutn = graph.dout_plus[u]+koutp, graph.dout_minus[u]+koutn
        din = dinp+dinn+epsilon
        dout = doutp+doutn+epsilon
        pinp, pinn, poutp, poutn = dinp/din, dinn/din, doutp/dout, doutn/dout
        nu, nv = graph.din[u] - (dinp+dinn), graph.dout[u] - (doutp+doutn)
        bpinp, bpinn = (dinp + prior_p*nu)/(nu + din), (dinn + prior_n*nu)/(nu + din),
        bpoutp, bpoutn  = (doutp + prior_p*nv)/(nv + dout), (doutn + prior_n*nv)/(nv + dout)
        node_type = (dinp>0)*8+(dinn>0)*4+(doutp>0)*2+(doutn>0)
        distrib = next_state_distrib(node_type, (nu, nv), (pinp, pinn, poutp, poutn))
        return distrib + [bpinp, bpinn, bpoutp, bpoutn]

    Xnodes = np.zeros((graph.order, 20))
    for u in graph.Gfull:
        Xnodes[u, :] = node_quantity((u, 0, 0, 0, 0))

    Xbayes = np.zeros((len(graph.E), 256+8+16))

    def compute_interactions(distrib_u, distrib_v):
        # return np.hstack([distrib_u, distrib_v])
        return np.kron(distrib_u, distrib_v)
    compute_interactions = np.kron

    test_edges_idx = set(test_set)
    sstart = clock()
    for (u, v), i in graph.edge_order.items():
        fu, fv = Xnodes[u, :], Xnodes[v, :]
        if i in test_edges_idx:
            dup = node_quantity((u, 0, 0, 1, 0))[:16]
            dun = node_quantity((u, 0, 0, 0, 1))[:16]
            dvp = node_quantity((v, 1, 0, 0, 0))[:16]
            dvn = node_quantity((v, 0, 1, 0, 0))[:16]
            interactions = prior_p*compute_interactions(dup, dvp) + prior_n*compute_interactions(dun, dvn)
            Xbayes[i, :] = np.hstack([interactions, fu[16:], fv[16:], Xa[i, 17:33]])
        else:
            Xbayes[i, :] = np.hstack([compute_interactions(fu[:16], fv[:16]), fu[16:], fv[16:], Xa[i, 17:33]])
    time_taken = (clock() - sstart)
    return Xbayes, time_taken


if __name__ == "__main__":
    import LillePrediction as llp
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, f1_score
    from exp_tworules import find_threshold

    graph = llp.LillePrediction(use_triads=True)
    graph.load_data('wik')
    nrep = 5
    res = np.zeros((nrep, 5))
    res_mcc = np.zeros((nrep, 5))
    for rep in range(5):
        es = graph.select_train_set(batch=.15)
        Xl, yl, train_set, test_set = graph.compute_features()
        Xa, ya = np.array(Xl), np.array(yl)
        Xbayes, time_taken = compute_bayes_features(Xa, ya, train_set, test_set, graph)

        logreg = SGDClassifier(loss='log', n_iter=5, class_weight={0: 1.4, 1: 1},
                               warm_start=True, average=True)
        sstart = clock()
        logreg.fit(Xbayes[train_set, :], ya[train_set])
        time_taken += clock() - sstart
        gold=ya[test_set]
        pred = logreg.predict(Xbayes[test_set, :])
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        res[rep, :] = ([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
                        matthews_corrcoef(gold, pred), fp/(fp+tn), time_taken])
        print(time_taken)
        Xtrain, ytrain = Xbayes[train_set, :], ya[train_set]
        Xtest, ytest = Xbayes[test_set, :], ya[test_set]

        feats_train = logreg.predict_proba(Xtrain)[:, 1]
        k = -find_threshold(-feats_train, ytrain, True)
        feats_test = logreg.predict_proba(Xtest)[:, 1]
        pred = feats_test > k
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        res_mcc[rep, :] = ([accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
                            matthews_corrcoef(gold, pred), fp/(fp+tn), time_taken])
    print(res.mean(0), res.std(0))
    print(res_mcc.mean(0), res_mcc.std(0))
