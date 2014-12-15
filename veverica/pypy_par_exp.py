#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import convert_experiment as cexp

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    from multiprocessing import Pool
    pool = Pool(cexp.NUM_THREADS)
    n = int(sys.argv[1])
    cexp.run_circle_experiment(n, one_at_a_time=True,
                               n_rep=1*cexp.NUM_THREADS, pool=pool)
    pool.close()
    pool.join()
