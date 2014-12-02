#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import experiments as xp

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    pool = Pool(10)
    for n_squares in [65, 82, 100]:
        xp.run_ring_experiment(2+2*n_squares, n_squares, n_rep=350, pool=pool)
    pool.close()
    pool.join()
