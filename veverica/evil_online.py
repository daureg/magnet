from heap.heap import heap
import graph_tool.generation as ggen
import pack_graph as pg
import random
import numpy as np


def create_topology(n=1000):
    G, E = pg.load_directed_signed_graph('directed_wik.pack')
    Wdout, Wdin = np.zeros(len(G)), np.zeros(len(G))
    for u, v in E:
        Wdout[u] += 1
        Wdin[v] += 1

    def degree_function():
        dout = random.choice(Wdout)*(n/len(G))+random.randint(1, 3)
        din = random.choice(Wdin)*(n/len(G))+random.randint(1, 3)
        return dout, din

    g = ggen.random_graph(n, degree_function)
    G, Gin, E = {}, {}, {}
    dout, din = np.zeros(n), np.zeros(n)
    for e in g.edges():
        u, v = int(e.source()), int(e.target())
        if u in G:
            G[u].add(v)
        else:
            G[u] = {v}
        if v in Gin:
            Gin[v].add(u)
        else:
            Gin[v] = {u}
        dout[u] += 1
        din[v] += 1
        E[(u, v)] = 1
    return G, Gin, E, dout, din, g


def initial_sign(n, G, E, dout, din):
    trollness = []
    for u in range(n):
        if random.random() > .7:
            trollness.append(random.uniform(.7, 1))
        else:
            trollness.append(random.uniform(0.05, .3))

    for u, nei in G.items():
        snei = sorted(nei, key=lambda v: trollness[v], reverse=True)
        t = trollness[u]
        nn = int(t*len(nei))
        for v in snei[:nn]:
            E[(u, v)] = -1
    doutm, dinm = np.zeros(n), np.zeros(n)
    for (u, v), s in E.items():
        if s < 0:
            doutm[u] += 1
            dinm[v] += 1

    def pij(i, j):
        return (doutm[i]+dinm[j])/(dout[i]+din[j])
    return trollness, doutm, dinm, pij


def refine_labeling(G, Gin, degrees, E, pij, trollness, n):
    (dout, din, doutm, dinm) = degrees
    increasing_dinm = np.argsort(dinm)
    increasing_doutm = np.argsort(doutm)

    def evaluate():
        undecided, correct = 0, 0
        for (u, v), s in E.items():
            p = pij(u, v)
            if abs(0.5 - p) < 1e-5:
                undecided += 1
            else:
                correct += int(s == np.sign(0.5 - p))
        return correct/len(E), undecided/len(E)

    def flip():
        for (u, v), s in E.items():
            pred = (np.sign(0.5 - pij(u, v)))
            if pred != 0 and pred != s:
                E[(u, v)] = -s

    def break_ties():
        nE = {}
        for (u, v), s in E.items():
            if abs(0.5 - pij(u, v)) > 1e-5:
                continue
            add_positive = s > 0
            add_outgoing = random.random() > .5
            if add_outgoing:
                nu = u
                if add_positive:
                    nv = random.choice(increasing_dinm[:int(n*.15)])
                else:
                    nv = random.choice(increasing_dinm[-int(n*.15):])
            else:
                nv = v
                if add_positive:
                    nu = random.choice(increasing_doutm[:int(n*.15)])
                else:
                    nu = random.choice(increasing_doutm[-int(n*.15):])
            ns = 1 if add_positive else -1
            G[nu].add(nv)
            Gin[nv].add(nu)
            nE[(nu, nv)] = ns
            dout[nu] += 1
            doutm[nu] += int(add_positive)
            din[nv] += 1
            dinm[nv] += int(add_positive)
        E.update(nE)

    for i in range(4):
        print(evaluate())
        flip()
        break_ties()


def my_rule(mi, oi, mj, oj, pos_frac):
    if oi+oj == 0:
        pij = 0.5
    else:
        pij = (mi+mj)/(oi+oj)
    if abs(pij - .5) > 1e-5:
        return 1 if pij < .5 else -1
    return 1 if random.random() < pos_frac else -1


def predict_fixed_order(n, E, edges):
    ep, em, houtm, hout = 0, 0, np.zeros(n), np.zeros(n)
    hinm, hin = np.zeros(n), np.zeros(n)
    pred, gold = [], []
    for u, v in edges:
        s = E[(u, v)]
        gold.append(s)
        p = my_rule(houtm[u], hout[u], hinm[v], hin[v],
                    0.5 if (ep+em) == 0 else ep/(ep+em))
        pred.append(p)
        hout[u] += 1
        houtm[u] += int(s < 0)
        hin[v] += 1
        hinm[v] += int(s < 0)
    return (np.array(pred) != np.array(gold)).cumsum()


def predict_evil_order(n, E, G, Gin, pij):
    ep, em, houtm, hout = 0, 0, np.zeros(n), np.zeros(n)
    hinm, hin = np.zeros(n), np.zeros(n)
    ref = {e: abs(0.5 - pij(*e)) for e in E}
    hpij = {e: 0 for e in E}
    pred, gold = [], []
    edges = heap({e: (hpij[e], ref[e]) for e in ref})
    while edges:
        u, v = edges.pop()
        s = E[(u, v)]
        gold.append(s)
        p = my_rule(houtm[u], hout[u], hinm[v], hin[v],
                    0.5 if (ep+em) == 0 else ep/(ep+em))
        pred.append(p)
        hout[u] += 1
        houtm[u] += int(s < 0)
        hin[v] += 1
        hinm[v] += int(s < 0)
        savev = v
        for v in G[u]:
            if (u, v) in edges:
                hpuv = (houtm[u] + hinm[v])/(hout[u] + hin[v])
                edges[(u, v)] = (abs(.5 - hpuv), ref[(u, v)])
        v = savev
        for u in Gin[v]:
            if (u, v) in edges:
                hpuv = (houtm[u] + hinm[v])/(hout[u] + hin[v])
                edges[(u, v)] = (abs(.5 - hpuv), ref[(u, v)])
    return (np.array(pred) != np.array(gold)).cumsum()


if __name__ == "__main__":
    # pylint: disable=C0103
    import matplotlib.pyplot as plt
    n = 1000
    G, Gin, E, dout, din, g = create_topology(n)
    trollness, doutm, dinm, pij = initial_sign(n, G, E, dout, din)
    refine_labeling(G, Gin, (dout, din, doutm, dinm), E, pij, trollness, n)
    k = 10
    histo = np.zeros((k, len(E)))
    edges = list(E.keys())
    for i in range(k):
        random.shuffle(edges)
        histo[i, :] = predict_fixed_order(n, E, edges)
    adaptive = predict_evil_order(n, E, G, Gin, pij)
    edges = sorted(E.keys(), key=lambda e: abs(0.5 - pij(*e)), reverse=False)
    fixed = predict_fixed_order(n, E, edges)
    plt.plot(np.mean(histo, 0), label='random')
    plt.plot(fixed, label='sorted')
    plt.plot(adaptive, label='adaptive')
    plt.legend(loc='lower right')
