#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""."""
import LinkPrediction as lp
import leskovec as l
from copy import deepcopy

class LillePrediction(lp.LinkPrediction):
    """My implementation of LinkPrediction"""

    def load_data(self, dataset, balanced=False):
        l.rw.read_original_graph(lp.FILENAMES[dataset], directed=True,
                                 balanced=balanced)
        Gfull, E = l.rw.G, l.rw.EDGE_SIGN
        self.order = len(Gfull)
        self.dout, self.din = l.defaultdict(int), l.defaultdict(int)
        for u, v in E:
            self.dout[u] += 1
            self.din[v] += 1
        self.common_nei = {e: Gfull[e[0]].intersection(Gfull[e[1]]) for e in E}
        self.Gout, self.Gin = {}, {}
        for u, v in E:
            l.add_neighbor(u, v, self.Gout)
            l.add_neighbor(v, u, self.Gin)
        self.G = self.Gout
        self.E = E

    def select_train_set(self, **params):
        if 'batch' in params:
            alpha = params['batch']*self.order/len(self.E)
            self.Esign = l.trolls.select_edges(None, self.E, alpha, 'random')
            return self.Esign
        else:
            alpha = params.get('alpha', 0)
            sf = params.get('sampling')
            Eout = l.trolls.select_edges(self.Gout, self.E, alpha, 'uniform', True, sf)
            Ein = l.trolls.select_edges(self.Gin, self.E, alpha, 'uniform', True, sf)
            directed_edges = deepcopy(Ein)
            directed_edges.update(Eout)
            self.Esign = directed_edges
            return directed_edges

    def compute_global_features(self):
        self.din_plus, self.dout_plus = l.defaultdict(int), l.defaultdict(int)
        self.din_minus, self.dout_minus = l.defaultdict(int), l.defaultdict(int)
        for (u, v), sign in self.Esign.items():
            if sign is True:
                self.din_plus[v] += 1
                self.dout_plus[u] += 1
            else:
                self.din_minus[v] += 1
                self.dout_minus[u] += 1

        
    def compute_one_edge_feature(self, edge):
        u, v = edge
        known_out = self.dout_plus[u]+self.dout_minus[u]
        known_in = self.din_plus[v]+self.din_minus[v]
        degrees = [self.dout[u], self.din[v], len(self.common_nei[(u, v)]),
                   self.din_plus[u], self.din_minus[v],
                   self.dout_plus[u], self.dout_minus[v],
                   self.din[u], self.dout[v],
                   self.din_plus[v], self.din_minus[u],
                   self.dout_plus[v], self.dout_minus[u],
                   self.dout_minus[u]/self.dout[u], self.din_minus[v]/self.din[v],
                   0 if known_out == 0 else self.dout_minus[u]/known_out,
                   0 if known_in == 0 else self.din_minus[v]/known_in,
                   ]
        triads = 16*[0, ]
        if self.with_triads:
            for w in self.common_nei[(u, v)]:
                for t in l.triads_indices(u, v, w, self.Esign):
                    triads[t] += 1
        return degrees+triads

if __name__ == '__main__':
    # pylint: disable=C0103
    graph = LillePrediction(use_triads=True)
    graph.load_data(lp.DATASETS.Wikipedia)
