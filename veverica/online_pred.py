from sklearn.linear_model import SGDClassifier
from timeit import default_timer as clock
import numpy as np
import random

def tree_prediction(features, cst, troll_first=True):
    if troll_first:
        return features[1] < (cst[2] + (features[0]<cst[0])*(cst[1]-cst[2]))
    return features[0] < (cst[2] + (features[1]<cst[0])*(cst[1]-cst[2]))

method_names=['only_troll_0.5', 'only_troll_transfer', 'only_pleas_0.5',
              'only_pleas_transfer', 'first_troll_.5', 'first_pleas_.5',
              'first_troll_transfer', 'first_pleas_transfer', 'perceptron',
              'lperceptron',  'dicho', 'cheat_dicho', 'allones', 'random',
              'majority vote',]

def online_prediction(graph, lambdas_t, lambdas_p, pp_start, alpha=1e-5):
    E = graph.E
    classes=np.array(sorted(set(graph.E.values())))
    din_plus, din_minus = {}, {}
    dout_plus, dout_minus = {}, {}
    trollness, unpleasantness = {}, {}
    positive_so_far = 0
    total_so_far = 0
    res = [[] for _ in method_names]
    edge_infos = list(E.items())
    random.shuffle(edge_infos)
    start= clock()
    Xa, ya = np.zeros((len(E), 2)), np.zeros(len(E))
    full_seen, last_partial = 0, 0
    fitted = False
    kd = 1
    class_weight={0:1.4, 1:1}
    perceptron = SGDClassifier(loss="perceptron", eta0=1, class_weight=class_weight,
                               learning_rate="constant", penalty=None, average=True, n_iter=4)
    lperceptron = SGDClassifier(loss="perceptron", eta0=1, class_weight=class_weight,
                               learning_rate="constant", penalty=None, average=True, n_iter=4)
    for (u, v), s in edge_infos:
        # get features
        known_out = dout_plus.get(u, 0)+dout_minus.get(u, 0)
        known_in = din_plus.get(v, 0)+din_minus.get(v, 0)
        trollness_u = None if known_out == 0 or u not in dout_minus else dout_minus[u]/known_out
        unpleasantness_v = None if known_in == 0 or v not in din_minus else din_minus[v]/known_in
        # predict if possible
        if trollness_u is not None:
            cst = [.5, 1, .0]
            res[0].append(int(s != tree_prediction((trollness_u, 0), cst, True)))
            cst = lambdas_t.copy(); cst[1] = 1; cst[2] = 0
            res[1].append(int(s != tree_prediction((trollness_u, 0), cst, True)))
        if unpleasantness_v is not None:
            cst = [.5, 1, .0]
            res[2].append(int(s != tree_prediction((0, unpleasantness_v), cst, False)))
            cst = lambdas_p.copy(); cst[1] = 1; cst[2] = 0
            res[3].append(int(s != tree_prediction((0, unpleasantness_v), cst, False)))
        if trollness_u is not None and unpleasantness_v is not None:
            cst = [.5, .5, .5]
            res[4].append(int(s != tree_prediction((trollness_u, unpleasantness_v), cst, True)))
            cst = [.5, .5, .5]
            res[5].append(int(s != tree_prediction((trollness_u, unpleasantness_v), cst, False)))
            cst = lambdas_t.copy()
            res[6].append(int(s != tree_prediction((trollness_u, unpleasantness_v), cst, True)))
            cst = lambdas_p.copy()
            res[7].append(int(s != tree_prediction((trollness_u, unpleasantness_v), cst, False)))
            Xa[full_seen, :], ya[full_seen] = (trollness_u, unpleasantness_v), int(s)
            if full_seen == pp_start:
                perceptron.fit(Xa[:full_seen, :], ya[:full_seen])
                lperceptron.fit(Xa[:full_seen, :], ya[:full_seen])
                lcoeff = lperceptron.coef_.ravel()
                lintercept = lperceptron.intercept_
                coeff = perceptron.coef_.ravel()
                intercept = perceptron.intercept_
                fitted = True
                last_partial = full_seen
            if full_seen - last_partial > 20:
                lperceptron.partial_fit(Xa[last_partial:full_seen, :], ya[last_partial:full_seen], classes)
                lcoeff = lperceptron.coef_.ravel()
                lintercept = lperceptron.intercept_
                last_partial = full_seen
            if fitted:
                res[8].append(int(s != (coeff[0]*trollness_u + coeff[1]*unpleasantness_v > -intercept)))
                res[9].append(int(s) != lperceptron.predict([[trollness_u, unpleasantness_v]]))
            x_i = trollness_u + unpleasantness_v
            pred = int(x_i < kd)
            if full_seen > pp_start:
                kd += alpha*((int(s) - pred))*x_i
            res[10].append(int(s) != pred)
            res[11].append(int(s != (x_i < 1)))
            full_seen += 1
        res[12].append(int(s != True))
        res[13].append(int(random.random() > .5))
        # otherwise toss a coin
        plus_proba = 0.5 if total_so_far < 10 else positive_so_far/total_so_far
        for method in res:
            if len(method) == total_so_far:
                method.append(int(random.random() > plus_proba))
        # update
        positive_so_far += int(s)
        total_so_far += 1
        if s:
            if v in din_plus:
                din_plus[v] += 1
            else:
                din_plus[v] = 1
            if u in dout_plus:
                dout_plus[u] += 1
            else:
                dout_plus[u] = 1
        else:
            if v in din_minus:
                din_minus[v] += 1
            else:
                din_minus[v] = 1
            if u in dout_minus:
                dout_minus[u] += 1
            else:
                dout_minus[u] = 1
    mistakes = np.array(res).sum(1)/len(E)
    print(kd)
    return mistakes

if __name__ == "__main__":
    import LillePrediction as llp
    data = {'WIK': llp.lp.DATASETS.Wikipedia,
            'EPI': llp.lp.DATASETS.Epinion,
            'SLA': llp.lp.DATASETS.Slashdot}
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices=data.keys(), default='WIK')
    parser.add_argument("-b", "--balanced", action='store_true',
                        help="Should there be 50/50 +/- edges")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=10)
    args = parser.parse_args()
    pref = args.data
    num_rep = args.nrep
    balanced = args.balanced
    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(data[pref], balanced=balanced)
    lambdas_t = llp.lambdas_troll[pref]
    lambdas_p = llp.lambdas_pleas[pref]
    pp_start = 500 if pref == 'WIK' else 4000
    alpha = 1e-5

    if args.balanced:
        pref += '_bal'
    start = (int(time.time()-(2015-1970)*365.25*24*60*60))//60
    fres = []
    for rep in range(num_rep):
        mistakes = online_prediction(graph, lambdas_t, lambdas_p, pp_start, alpha)
        fres.append(mistakes)
        for i in np.argsort(mistakes)[:1]:
            print('{} {:,}'.format(method_names[i].ljust(25), (mistakes[i])))
    np.savez_compressed('{}_online_{}.npz'.format(pref, start), res=np.array(fres))
