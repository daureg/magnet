# vim: set fileencoding=utf-8
"""."""
from enum import Enum
from timeit import default_timer as clock
import numpy as np
from collections import defaultdict
DATASETS = Enum('Dataset', 'Wikipedia Slashdot Epinion WikEdits')
FILENAMES = {DATASETS.Wikipedia: 'soc-wiki.txt',
             DATASETS.Slashdot: 'soc-sign-Slashdot090221.txt',
             DATASETS.Epinion: 'soc-sign-epinions.txt',
             DATASETS.WikEdits: 'soc-sign-kiw.txt'}
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

class LinkPrediction(object):
    """
    Base class that hold a signed graph and allow various method of predicting
    its sign
    
    Attributes:
        G A dictionnary where keys are nodes and values are outgoing neighbors
        E A dictionnary where keys are edges and values are boolean sign
    """

    def __init__(self, use_triads=False):
        self.G, self.E = {}, {}
        self.with_triads = use_triads
        self.time_used = 0
        
    def load_data(self, dataset, balanced=False):
        """Load one of the 3 dataset to fill G and E"""
        pass

    def select_train_set(self, *params):
        """Return a subset of `E` from which sign will be known"""
        pass

    def compute_global_features(self, selected_edges):
        """Compute side info to help with edge feature"""
        pass

    def compute_one_edge_feature(self, edge):
        """Compute a feature vector for a given edge"""
        pass

    def compute_features(self):
        self.compute_global_features()
        self.edge_order = {e: i for i, e in enumerate(sorted(self.E))}
        knows_indices, pred_indices = [], []
        features, signs = [], []
        for i, ((u, v), sign) in enumerate(sorted(self.E.items())):
            feature = self.compute_one_edge_feature((u, v))
            features.append(feature)
            signs.append(int(sign))
            (knows_indices if (u, v) in self.Esign else pred_indices).append(i)
        self.features = np.array(features)
        return features, signs, knows_indices, pred_indices

    def get_partial_adjacency(self):
        """return a dict of dict simulating a the symetric adjacency matrix of
        the observed network"""
        A = defaultdict(dict)
        for (u, v), sign in self.Esign.items():
            # in the small case of conflict, last seen sign wins
            A[u][v] = 2*int(sign) - 1
            A[v][u] = 2*int(sign) - 1
        return A

    def train(self, model, *args):
        self.time_used = 0
        if hasattr(model, 'fit'):
            s = clock()
            model.fit(*args)
            self.time_used += clock() - s
            return model.predict
        return model

    def test_and_evaluate(self, pred_function, pred_data, gold,
                          postprocess=None, only_on_lcc=False):
        from timeit import default_timer as clock
        frac = 1-gold.size/len(self.E)
        if only_on_lcc:
            frac = 1-gold.size/self.in_lcc.sum()
        s = clock()
        pred = pred_function(pred_data)
        self.time_used += clock() - s
        if postprocess:
            test_set, idx2edge = postprocess
            pred_reciprocal = []
            for e, p in zip(test_set, pred):
                er = self.reciprocal.get(e)
                if er is None or idx2edge[er] not in self.Esign:
                    pred_reciprocal.append(p)
                else:
                    twin_sign = self.Esign[idx2edge[er]]
                    pred_reciprocal.append(twin_sign)
            pred = pred_reciprocal
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        return [accuracy_score(gold, pred), f1_score(gold, pred, average='weighted', pos_label=None),
                matthews_corrcoef(gold, pred), fp/(fp+tn), self.time_used, frac]

    # def online_mode(self, edge):
    #     # partial update of global feature and all edges including u and v
    #     pass
