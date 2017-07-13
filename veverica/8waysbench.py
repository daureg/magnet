# coding: utf-8
import warnings

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.simplefilter('ignore', FutureWarning)
RdGr = matplotlib.colors.LinearSegmentedColormap.from_list('RdGr', [matplotlib.colors.hex2color('#dd2c00'),
                                                                    matplotlib.colors.hex2color('#64dd17')], 2)
FULL = False
def plot_boundary(predict_fun, dataset, method):
    plot_step = .002
    xx, yy = np.meshgrid(np.arange(0,1, plot_step),
                         np.arange(0,1, plot_step))
    Z = predict_fun(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    size = (26/8)*2.54
    with sns.plotting_context(rc={'figure.figsize': (size,size)}):
        fig, ax1 = plt.subplots()
        ax1.contourf(xx, yy, Z, 2, cmap=RdGr)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        plt.axis('equal')
        plt.tick_params(axis='both', which='both', bottom='off', top='off',
                        left='off', labelbottom='off', labelleft='off')
        plt.show()
        plt.savefig('{}_{}{}.png'.format(dataset, method, '_full' if FULL else ''), dpi=300, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    import numpy as np
    import argparse
    import LillePrediction as llp
    data = {'WIK': llp.lp.DATASETS.Wikipedia,
            'EPI': llp.lp.DATASETS.Epinion,
            'SLA': llp.lp.DATASETS.Slashdot}
    alphas = [{'WIK': lambda d: int(ceil(.295*d)), 'SLA': lambda d: int(ceil(.278*d)), 'EPI': lambda d: int(ceil(.261*d))},
              {'WIK': lambda d: int(ceil(1*log(d))), 'SLA': lambda d: int(ceil(1*log(d))), 'EPI': lambda d: int(ceil(1*log(d)))},
              {'WIK': lambda d: int(ceil(2*log(d))), 'SLA': lambda d: int(ceil(2*log(d))), 'EPI': lambda d: int(ceil(2*log(d)))},
              {'WIK': lambda d: int(ceil(.076*d)), 'SLA': lambda d: int(ceil(.108*d)), 'EPI': lambda d: int(ceil(.082*d))},
              {'WIK': lambda d: int(ceil(.149*d)), 'SLA': lambda d: int(ceil(.215*d)), 'EPI': lambda d: int(ceil(.165*d))},
              {'WIK': lambda d: int(ceil(.076*d)), 'SLA': lambda d: int(ceil(.108*d)), 'EPI': lambda d: int(ceil(.082*d))},
              {'WIK': lambda d: int(ceil(.149*d)), 'SLA': lambda d: int(ceil(.215*d)), 'EPI': lambda d: int(ceil(.165*d))}
              ]
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Which data to use",
                        choices=data.keys(), default='WIK')
    parser.add_argument("-f", "--full", action='store_true',
                        help="Use full dataset for training")
    parser.add_argument("-n", "--nrep", help="number of repetition", type=int,
                        default=10)
    parser.add_argument("-s", "--sampling", help="Sampling scheme", type=int,
                        choices=list(range(len(alphas))), default=0)
    args = parser.parse_args()
    pref = args.data
    nrep = args.nrep
    smpsch = args.sampling
    from math import ceil, log
    if args.full:
        FULL = True
        for k in alphas[0]:
            alphas[0][k] += 1
    graph = llp.LillePrediction(use_triads=False)
    graph.load_data(data[pref], balanced=False)
    import time
    start = (int(time.time()-(2016-1970)*365.25*24*60*60))//60

    from sklearn.linear_model import SGDClassifier
    from sklearn.tree import DecisionTreeClassifier
    from QuadrantClassifier import QuadrantClassifier
    from adhoc_DT import AdhocDecisionTree
    cw={0: 1.4, 1: 1}
    onedt_fixed = AdhocDecisionTree(troll_first=True)
    onedt_fixed.threshold = [.5, .5, .5]
    onedt_fixed.decision = [1, 1, 0, 0]
    onedt_learned = AdhocDecisionTree(troll_first=True)
    twodt_fixed = AdhocDecisionTree(troll_first=False)
    twodt_fixed.threshold = [.5, .5, .5]
    twodt_fixed.decision = [1, 0, 1, 0]
    twodt_learned = AdhocDecisionTree(troll_first=False)
    pa = SGDClassifier(loss="perceptron", eta0=1, n_iter=4, class_weight=cw, learning_rate="constant", penalty=None, average=True)
    aggregate = lambda X: X[:,0] < 1-X[:,1]
    fabio = QuadrantClassifier(sub_strategy='perceptron', lambdas=[.5,.5,.5], troll_first=True, Fabio_bias=True, inner_classifier=None)
    inner = SGDClassifier(class_weight=cw, learning_rate="optimal", average=True, n_iter=4, loss='hinge', penalty='l2')
    myself = QuadrantClassifier(sub_strategy='perceptron', lambdas=None, troll_first=False, inner_classifier=inner, Fabio_bias=False)
    from sklearn.kernel_approximation import Nystroem
    from sklearn.pipeline import make_pipeline
    rbf_pa = make_pipeline(Nystroem(gamma=31, kernel_params={'C': 600}, n_components=80),
                           SGDClassifier(loss="perceptron", eta0=1, n_iter=4, class_weight=cw,
                                         learning_rate="constant", penalty=None, average=True))
    from L1Classifier import L1Classifier
    simple = L1Classifier()
    tweaked_aggregate = lambda x: x[1][:,0]*x[0][:,0] < x[1][:,1]*(1-x[0][:,1])
    circle_rule = lambda X:(X[:,0]-1)**2 + (X[:, 1]-1)**2 > 1

    fres = [[] for _ in range(12)]
    only_t_fixed, only_t_learned = [], []
    first_p_fixed, first_p_learned, percep = [], [], []
    new_rule, quadrant, quadrant_me, gaussian = [], [], [], []
    dichotomy_rule, weighted_rule, circle = [], [], []
    for _ in range(nrep):
        es=graph.select_train_set(sampling=alphas[smpsch][pref],
                                  with_replacement=smpsch<5)
        print(100*len(es)/len(graph.E))
        Xl, yl, train_set, test_set = graph.compute_features()
        Xa, ya = np.array(Xl)[:, 15:17], np.array(yl)
        if args.full:
            train_set = np.arange(Xa.shape[0])
            test_set = train_set
        gold=ya[test_set]
        pred_function = graph.train(lambda X: onedt_fixed.predict(X))
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'only_t_fixed')
        only_t_fixed.append(res)
        pred_function = graph.train(onedt_learned, Xa[train_set, :], ya[train_set])
        onedt_learned.decision = [1, 1, 0, 0]
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'only_t_learned')
        only_t_learned.append(res)
        pred_function = graph.train(lambda X: twodt_fixed.predict(X))
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'first_p_fixed')
        first_p_fixed.append(res)
        pred_function = graph.train(twodt_learned, Xa[train_set, :], ya[train_set])
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'first_p_learned')
        first_p_learned.append(res)
        pred_function = graph.train(pa, Xa[train_set, :], ya[train_set])
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'percep')
        percep.append(res)
        pred_function = graph.train(aggregate)
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'new_rule')
        new_rule.append(res)
        pred_function = graph.train(fabio, Xa[train_set, :], ya[train_set])
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'quadrant')
        quadrant.append(res)
        pred_function = graph.train(myself, Xa[train_set, :], ya[train_set])
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'quadrant_me')
        quadrant_me.append(res)
        pred_function = graph.train(simple, Xa[train_set, :], ya[train_set])
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'dichotomy_rule')
        dichotomy_rule.append(res)
        nSamp = np.array(Xl)[:, -2:]
        samples = 1-np.exp(-nSamp/2)
        pred_function = graph.train(tweaked_aggregate)
        res = graph.test_and_evaluate(pred_function, (Xa[test_set, :], samples[test_set, :]), gold)
        weighted_rule.append(res)
        pred_function = graph.train(circle_rule)
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'circle')
        circle.append(res)
        # gaussian.append([.7,.7,.5,.3,5,.5])
        # continue
        pred_function = graph.train(rbf_pa, Xa[train_set, :], ya[train_set])
        res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        if _==0:
            plot_boundary(pred_function, pref, 'gaussian')
        gaussian.append(res)
    fres[0].append(only_t_fixed)
    fres[1].append(only_t_learned)
    fres[2].append(first_p_fixed)
    fres[3].append(first_p_learned)
    fres[4].append(percep)
    fres[5].append(new_rule)
    fres[6].append(dichotomy_rule)
    fres[7].append(circle)
    fres[8].append(weighted_rule)
    fres[9].append(quadrant)
    fres[10].append(quadrant_me)
    fres[11].append(gaussian)
    np.savez_compressed('{}{}_8ways_{}'.format(pref, '_full' if args.full else '', start), res=np.array(fres))
