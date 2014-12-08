#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import experiments as xp
from math import ceil, sqrt

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    pool = Pool(xp.NUM_THREADS)
    n = 40
    p_strategies = [xp.densify.PivotStrategy.uniform,
                    xp.densify.PivotStrategy.no_pivot]
    t_strategy = [xp.TriangleStatus.closeable,
                  xp.TriangleStatus.any,
                  xp.TriangleStatus.one_edge_missing,
                  xp.TriangleStatus.one_edge_positive]
    strats = [(p_strategies[1], t_strategy[0])]
    strats.extend([(p_strategies[0], _) for _ in t_strategy])
    for strategy in strats:
        for distance in [n//4, n//2, 2, 1]:
            negative_edges = xp.negative_pattern(n, distance=distance)
            xp.run_circle_experiment(n, pivot_strategy=strategy[0],
                                     triangle_strategy=strategy[1],
                                     rigged=negative_edges, pool=pool)
        negative_edges = xp.negative_pattern(n, quantity=ceil(sqrt(n)))
        xp.run_circle_experiment(n, pivot_strategy=strategy[0],
                                 triangle_strategy=strategy[1],
                                 rigged=negative_edges, pool=pool)
    xp.run_circle_experiment(n, one_at_a_time=True, pool=pool)
    xp.run_circle_experiment(n, one_at_a_time=False, pool=pool)
