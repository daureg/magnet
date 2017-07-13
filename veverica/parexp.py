#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
from itertools import product

import experiments as xp

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    from multiprocessing import Pool
    pool = Pool(xp.NUM_THREADS)
    n = 70
    p_strategies = [xp.densify.PivotStrategy.uniform,
                    xp.densify.PivotStrategy.no_pivot]
    t_strategy = [xp.TriangleStatus.closeable,
                  xp.TriangleStatus.any,
                  xp.TriangleStatus.one_edge_missing,
                  xp.TriangleStatus.one_edge_positive]
    strats = [(p_strategies[1], t_strategy[0])]
    strats.extend([(p_strategies[0], _) for _ in t_strategy])
    # n = int(sys.argv[1])
    for (strategy, d) in product(strats, [1, 2]):
        xp.run_circle_experiment(n, pivot_strategy=strategy[0],
                                 rigged=xp.negative_pattern(n, dash_length=d),
                                 n_rep=6*xp.NUM_THREADS,
                                 triangle_strategy=strategy[1], pool=pool)
    # n_squares = int(sys.argv[1])
    # for strategy in strats:
    #     xp.run_ring_experiment(2+2*n_squares, n_squares,
    #                            pivot_strategy=strategy[0], n_rep=50,
    #                            triangle_strategy=strategy[1], pool=pool)
    pool.close()
    pool.join()
