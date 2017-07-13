import random
import sys
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm, trange

NUM_THREADS = 14
CHUNK_SIZE = 5


def mistakes(preds, gold):
    return (np.sign(preds.sum(0)) != gold).sum()/gold.size


def trees_are_random(filename):
    res_file = filename.replace('_0', '_processed')
    with np.load(filename) as f:
        res, gold = f['res'], f['gold']
    num_trees, num_order, _ = res.shape
    all_trees = list(range(num_trees))
    all_order = list(range(num_order))
    all_pred = np.arange(len(gold))
    nrep = 100
    rate = np.zeros((num_trees//2+1, [j//13-1 for j in range(13, num_order, 13)][-1]+1, 2))
    frac = 1.3
    dropped = []
    for i in trange(1, num_trees+1, 2):
        for j in trange(13, num_order, 13):
            tmp_res = []
            for k in range(nrep):
                trees = random.sample(all_trees, i)
                orders = random.sample(all_order, j-1 if j % 2 == 0 else j)
                if i == 1:
                    vals = res[trees, orders, :]
                else:
                    vals = res[np.ix_(trees, orders, all_pred)].sum(0)
                tmp_res.append(mistakes(vals))
            thre = frac*np.median(tmp_res)
            good = tmp_res < thre
            bad = np.logical_not(good)
            dropped.append((i, j, bad.sum()/nrep))
            rate[(i-1)//2, j//13-1, :] = np.mean(tmp_res), np.std(tmp_res)
            np.savez_compressed(res_file, rate=rate)


def closed_odd(x):
    as_int = int(np.round(x))
    if as_int % 2 == 1:
        return as_int
    before, after = as_int - 1, as_int + 1
    if x - before < after - x:
        return before
    return after


def more_random_trees(filename):
    res_file = filename.replace('_0', '_more_random')
    with np.load(filename) as f:
        res, gold = f['res'], f['gold']
    num_trees, num_order, _ = res.shape
    all_trees = list(range(num_trees))
    nrep_tree, nrep_order = 20, 30
    n_trees = np.arange(1, 28, 4)
    n_trees[-1] += 2
    n_orders = np.array([closed_odd(x) for x in np.linspace(13, 13*9, 26)])
    rate = np.zeros((len(n_trees), len(n_orders), 2))
    for i, n_tree in enumerate(tqdm(n_trees)):
        for j, n_order in enumerate(tqdm(n_orders)):
            tmp_res = []
            orders = np.random.randint(0, num_order, (nrep_order, n_order))
            for k in range(nrep_tree):
                trees = random.sample(all_trees, n_tree)
                vals = res[trees, :, :].sum(0)
                for order in orders:
                    tmp_res.append(mistakes(vals[order, :], gold))
            rate[i, j, :] = np.mean(tmp_res), np.std(tmp_res)
            np.savez_compressed(res_file, rate=rate)


def trees_are_fixed(filename):
    res_file = filename.replace('_0', '_fixedtree')
    with np.load(filename) as f:
        res, gold = f['res'], f['gold']
    num_trees, num_order, _ = res.shape
    all_order = list(range(num_order))
    n_trees = list(range(1, num_trees+1, 2))
    n_orders = [j-1 if j % 2 == 0 else j for j in range(13, num_order, 13)]
    nrep = NUM_THREADS*CHUNK_SIZE
    rate = np.zeros((len(n_trees), len(n_orders), 2))
    pool = Pool(NUM_THREADS)
    for i, n_tree in enumerate(tqdm(n_trees)):
        for j, n_order in enumerate(tqdm(n_orders)):
            args = zip(repeat(all_order, nrep), repeat(n_order, nrep),
                       repeat(n_tree, nrep), repeat(res, nrep), repeat(gold, nrep))
            tmp_res = list(pool.imap_unordered(run_fixed, args, CHUNK_SIZE))
            rate[i, j, :] = np.mean(tmp_res), np.std(tmp_res)
            np.savez_compressed(res_file, rate=rate)
    pool.close()
    pool.join()


def run_fixed(args):
    all_order, n_order, n_tree, res, gold = args
    orders = random.sample(all_order, n_order)
    return mistakes(res[:n_tree, orders, :].sum(0), gold)


if __name__ == "__main__":
    filename = sys.argv[1]
    more_random_trees(filename)
