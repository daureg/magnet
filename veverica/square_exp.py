#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import os
import sys

import convert_experiment as cexp

sys.path.append(os.path.expanduser('~/venvs/34/lib/python3.4/site-packages/'))

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    NUM_THREADS = 14
    cexp.NUM_THREADS = NUM_THREADS
    pool = Pool(NUM_THREADS)
    kind, n = int(sys.argv[1]), int(sys.argv[2])
    strategies = [cexp.redensify.PivotSelection.Uniform,
                  cexp.redensify.PivotSelection.Preferential,
                  cexp.redensify.PivotSelection.ByDegree]
    for s in strategies:
        if kind == 0:
            cexp.run_rings_experiment(n*n, n, pivot=s, shared_sign=True,
                                      rigged=False, one_at_a_time=True,
                                      n_rep=4*NUM_THREADS, pool=pool)
        if kind == 1:
            cexp.run_rings_experiment(2+2*n, n, pivot=s, shared_sign=True,
                                      rigged=False, one_at_a_time=True,
                                      n_rep=4*NUM_THREADS, pool=pool)
        if kind == 2:
            cexp.run_circle_experiment(n, one_at_a_time=True, rigged=False,
                                       n_rep=4*NUM_THREADS, pivot=s, pool=pool)
        if kind == 3:
            params = [(15, 5), (6, 30)][n]
            cexp.run_planted_experiment(params[0], params[1],
                                        one_at_a_time=True, pool=pool,
                                        n_rep=4*NUM_THREADS, pivot=s)
    pool.close()
    pool.join()
