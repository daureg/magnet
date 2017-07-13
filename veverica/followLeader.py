import random

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

import exp_tworules as etr
import LillePrediction as llp


def choose_k(X, ya, mcc=False):
    if not mcc:
        return .5, .5
    k_troll = etr.find_threshold(X[:,15], ya, mcc=True)
    k_pleas = etr.find_threshold(X[:,16], ya, mcc=True)
    return k_troll, k_pleas


def predict_supervised(X, ya, k_troll, k_pleas):
    pred=[]
    results = {'decided by trollness': 0,
               'decided by unpleasantness': 0,
               'undecided': 0,}
    for (t, u), s in zip(X[:, 15:17], ya):
        ptroll, ppleas = None, None
        if t > k_troll: ptroll = 0
        if t < k_troll: ptroll = 1
        if ptroll == s:
            pred.append(ptroll)
            results['decided by trollness'] += 1
            continue
        if u > k_pleas: ppleas = 0
        if u < k_pleas: ppleas = 1    
        if ppleas == s:
            pred.append(ppleas)
            results['decided by unpleasantness'] += 1
            continue
        results['undecided'] += 1
        pred.append(1-s)
    # print(k_troll, k_pleas, results)
    return pred

def eval_results(gold, pred):
    C = confusion_matrix(gold, pred)
    fp, tn = C[0, 1], C[0, 0]
    return (accuracy_score(gold, pred),
            f1_score(gold, pred, average='weighted', pos_label=None),
            matthews_corrcoef(gold, pred), fp/(fp+tn))


def predict_unsupervised(X, ya, k_troll, k_pleas):
    pred=[]
    results = {'decided by trollness': 0,
               'decided by unpleasantness': 0,
               'undecided': 0,
              'consensus': 0}
    for (t, u), s in zip(X[:, 15:17], ya):
        ptroll, ppleas = None, None
        if t > k_troll: ptroll = 0
        if t < k_troll: ptroll = 1
        if u > k_pleas: ppleas = 0
        if u < k_pleas: ppleas = 1
        if ptroll == ppleas:
            if ptroll is not None:
                results['consensus'] += 1
                pred.append(ptroll)
                continue
            results['undecided'] += 1
            pred.append(int(random.random()<.8))
            continue
        if ppleas is None:
            pred.append(ptroll)
            results['decided by trollness'] += 1
            continue
        if ptroll is None:
            pred.append(ppleas)
            results['decided by unpleasantness'] += 1
            continue
        results['undecided'] += 1
        pred.append(random.choice([ptroll, ppleas]))
    # print(k_troll, k_pleas, results)
    return pred

if __name__ == "__main__":
    from itertools import product
    graph=llp.LillePrediction(use_triads=False)

    using_mcc = [False, True]
    using_supervision = [predict_supervised, predict_unsupervised]
    batch_level = [1.0,]
    dataset = ['wik', 'sla', 'epi', 'kiw']
    for data in dataset:
        graph.load_data(data)
        for batch in batch_level:
            es=graph.select_train_set(batch=batch)

            Xl, yl, train_set, test_set = graph.compute_features()
            X, ya = np.array(Xl), np.array(yl)
            if len(test_set) < 10:
                test_set = train_set
                np.savez_compressed('full_{}'.format(data), X=X, y=ya)
            gold = ya[test_set]
            for mcc, pred_function in product(using_mcc, using_supervision):
                msg_args = [data.upper(), '' if mcc else 'not ']
                msg_args.append(pred_function.__name__.split('_')[-1])
                msg_args.append(100*batch)
                print('{}: {}MCC with {} at {:.1f}%'.format(*msg_args))


                pred = pred_function(X[test_set,:], ya[test_set],
                                     *choose_k(X[train_set, :], ya[train_set], mcc))
                print('\t'.join(map(lambda x: '{:.3f}'.format(x), eval_results(gold, pred))))
