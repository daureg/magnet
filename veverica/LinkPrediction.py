# vim: set fileencoding=utf-8
"""."""
from enum import Enum
DATASETS = Enum('Dataset', 'Wikipedia Slashdot Epinion')
FILENAMES = {DATASETS.Wikipedia: 'soc-wiki.txt',
             DATASETS.Slashdot: 'soc-sign-epinions.txt',
             DATASETS.Epinion: 'soc-sign-Slashdot090221.txt'}
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
        knows_indices, pred_indices = [], []
        features, signs = [], []
        for i, ((u, v), sign) in enumerate(self.E.items()):
            feature = self.compute_one_edge_feature((u, v))
            features.append(feature)
            signs.append(int(sign))
            (knows_indices if (u, v) in self.Esign else pred_indices).append(i)
        return features, signs, knows_indices, pred_indices

    def get_adjacency_matrix(self):
        # TODO: sparse matrix?
        import numpy
        n = len(self.G)
        A = np.zeros((n, n), int)
        for i, adj in self.G.items():
            js = sorted(adj)
            signs = [2*E[(i, j)]-1 for j in js]
            A[i, js] = signs
        return A

    def train(self, model, *args):
        if hasattr(model, 'fit'):
            model.fit(args)
            return model.predict
        return model

    def test_and_evaluate(self, pred_function, pred_data, gold):
        from timeit import default_timer as clock
        frac = 1.0 - len(pred_data)/len(E)
        s = clock()
        pred = pred_function(pred_data)
        end = clock()
        C = confusion_matrix(gold, pred)
        fp, tn = C[0, 1], C[0, 0]
        return [accuracy_score(gold, pred), f1_score(gold, pred),
                matthews_corrcoef(gold, pred), fp/(fp+tn), end-s, frac]

    # def online_mode(self, edge):
    #     # partial update of global feature and all edges including u and v
    #     pass
