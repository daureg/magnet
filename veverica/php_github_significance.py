# coding: utf-8
from collections import Counter
import persistent as p
import random
import influence
from itertools import combinations
NREP = 200
DM_over_PHP = False
prom = None


def switch_labels(labels, G, corrupted):
    """Set some nodes to 1 in order to reduce the number of errors""" 
    newlabels = labels.copy()
    posneg = {}
    for u in G:
        num_pos_neigbors = len({v for v in G[u] if newlabels[v] == 1})
        posneg[u] = (num_pos_neigbors, len(G[u]) - num_pos_neigbors)
    # It's working OK with nn==0 but it should be np==0
    no_one_neigh = {u for u, (np, nn) in posneg.items() if nn == 0}
    flipped = 0
    for u in corrupted[2][2]:
        neg_nei = {v for v in G[u] if newlabels[v] == 0}
        if all(v in no_one_neigh for v in neg_nei):
            flipped += 1
            newlabels[u] = 1
    print('flipped {} labels to 1'.format(flipped))
    return newlabels


def pairs_with_lot_of_common_neighbors(G):
    degrees = sorted(((node, len(adj)) for node, adj in G.items()),
                     key=lambda x: x[1])
    lot_common_neighbors = {nodes: G[nodes[0]].intersection(G[nodes[1]])
                            for nodes in combinations((_[0] for _ in degrees[-600:]), 2)}
    return {e: len(s) for e, s in lot_common_neighbors.items() if len(s) > 40}


def synthetic_label(G, E, nb_ones_target, num_seeds=5, noise=0.1):
    #prom = pairs_with_lot_of_common_neighbors(G)
    labels = {u: 0 for u in G}
    rts = random.sample(list(prom.keys()), num_seeds)
    before = 0
    for roots in rts:
        nb_ones = 0
        u, v = roots
        seen = set(roots)
        border, next_border = G[u].intersection(G[v]), set()
        labels[u] = 1
        labels[v] = 1
        for w in border:
            labels[w] = 1
            nb_ones += 1
        nb_iter = 0
        done = False
        while border and not done and nb_iter<len(G):
            nb_iter += 1
            seen.update(border)
            border = list(border)
            random.shuffle(border)
            for u in border:
                for v in G[u].difference(seen):
                    for w in G[v].intersection(seen):
                        if w == u:
                            continue
                        next_border.add(v)       
                        if random.random() > noise:
                            labels[v] = 1
                            nb_ones += 1
                    if (nb_ones - before) > nb_ones_target//len(rts):                    
                        done = True
                        break
                if done:
                    break
            border, next_border = next_border, set()
        before = nb_ones
    return labels

if __name__ == "__main__":
    if DM_over_PHP:
        fG, fE = p.load_var('citations_triangle_graph.my')
        from DM_labels import labels
    else:
        fG, fE = p.load_var('github_triangle_graph.my')
        from php_labels import labels
    prom = pairs_with_lot_of_common_neighbors(fG)
    gold = Counter(labels.values())
    print(gold)
    corrupted=influence.find_corrupted(fG, labels)
    k=2
    vals = corrupted[k]
    size = list(map(len, vals))
    us = 100.0*(size[1]+size[2])/size[0]
    print('{}: {} ({:.3f}%)'.format(k, size, us))
    random_labels = []
    for _ in range(NREP):
        labels = {u: random.random()<1.0*gold[True]/len(fG) for u in fG}
        corrupted=influence.find_corrupted(fG, labels, 2)
        nlabels = switch_labels(labels, fG, corrupted)
        corrupted=influence.find_corrupted(fG, nlabels, 2)
        k=2
        vals = corrupted[k]
        size = list(map(len, vals))
        err = 100.0*(size[1]+size[2])/size[0]
        print('{}: {} ({:.3f}%)'.format(k, size, err))
        random_labels.append(err)
    noise_labels = []
    noise_levels = [0, .1]
    for noise in noise_levels:
        synth_labels = []
        for _ in range(NREP):
            slabels = synthetic_label(fG, fE, gold[True], 3, noise)
            corrupted=influence.find_corrupted(fG, slabels, 2)
            nlabels = switch_labels(slabels, fG, corrupted)
            corrupted=influence.find_corrupted(fG, nlabels, 2)
            k=2
            vals = corrupted[k]
            size = list(map(len, vals))
            err = 100.0*(size[1]+size[2])/size[0]
            print('{}: {} ({:.3f}%)'.format(k, size, err))
            synth_labels.append(err)
        noise_labels.append(list(synth_labels))
    if DM_over_PHP:
        p.save_var('DM_label_res.my', (us, random_labels, noise_labels, noise_levels))
    else:
        p.save_var('php_label_res.my', (us, random_labels, noise_labels, noise_levels))
