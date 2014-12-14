#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import convert_experiment as cexp

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    from multiprocessing import Pool
    pool = Pool(cexp.NUM_THREADS)
    for k in range(2, 13):
        cexp.run_circle_experiment(2**k, one_at_a_time=True,
                                    n_rep=4*cexp.NUM_THREADS, pool=pool)
    pool.close()
    pool.join()

