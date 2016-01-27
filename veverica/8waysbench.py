# coding: utf-8
import warnings
warnings.simplefilter('ignore', FutureWarning)
import matplotlib
import matplotlib.pyplot as plt
RdGr = matplotlib.colors.LinearSegmentedColormap.from_list('RdGr', [matplotlib.colors.hex2color('#dd2c00'), 
                                                                    matplotlib.colors.hex2color('#64dd17')], 2)
import seaborn as sns
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
        plt.tick_params( axis='both', which='both', bottom='off', top='off',
                        left='off', labelbottom='off', labelleft='off')
        plt.show()
        plt.savefig('{}_{}.png'.format(dataset, method), dpi=300, bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    import numpy as np
    import LillePrediction as llp
    data = {'WIK': llp.lp.DATASETS.Wikipedia,
            'EPI': llp.lp.DATASETS.Epinion,
            'SLA': llp.lp.DATASETS.Slashdot}
    alphas = {'SLA': .278, 
              'WIK': .295,
              'EPI': .261}
    import sys
    pref = sys.argv[1]
    nrep = int(sys.argv[2])
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
                  SGDClassifier(loss="perceptron", eta0=1, n_iter=4, class_weight=cw, learning_rate="constant",
                                penalty=None, average=True))
    from L1Classifier import L1Classifier
    simple = L1Classifier()

    fres = [[] for _ in range(10)]
    only_t_fixed, only_t_learned = [], []
    first_p_fixed, first_p_learned, percep = [], [], []
    new_rule, quadrant, quadrant_me, gaussian = [], [], [], []
    dichotomy_rule = []
    for _ in range(nrep):
        es=graph.select_train_set(sampling=lambda d: int(alphas[pref]*d))
        print(100*len(es)/len(graph.E))
        Xl, yl, train_set, test_set = graph.compute_features()
        Xa, ya = np.array(Xl)[:, 15:17], np.array(yl)
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
        # pred_function = graph.train(rbf_pa, Xa[train_set, :], ya[train_set])
        # res = graph.test_and_evaluate(pred_function, Xa[test_set, :], gold)
        # if _==0:
        #     plot_boundary(pred_function, pref, 'gaussian')
        gaussian.append([.8,.8,.5,.3,.2,.5])
    fres[0].append(only_t_fixed)
    fres[1].append(only_t_learned)
    fres[2].append(first_p_fixed)
    fres[3].append(first_p_learned)
    fres[4].append(percep)
    fres[5].append(new_rule)
    fres[6].append(quadrant)
    fres[7].append(quadrant_me)
    fres[8].append(gaussian)
    fres[9].append(dichotomy_rule)
    np.savez_compressed('{}_8ways_{}'.format(pref, start), res=np.array(fres))
