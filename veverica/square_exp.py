#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import convert_experiment as cexp

if __name__ == '__main__':
    # pylint: disable=C0103
    import sys
    from multiprocessing import Pool
    NUM_THREADS = 14
    cexp.NUM_THREADS = NUM_THREADS
    pool = Pool(NUM_THREADS)
    n = int(sys.argv[1])
    params = [(15, 5)]
    cexp.run_planted_experiment(params[n][0], params[n][1],
                                one_at_a_time=True, n_rep=4*NUM_THREADS,
                                pool=pool)
    cexp.run_planted_experiment(params[n][0], params[n][1],
                                one_at_a_time=False, n_rep=4*NUM_THREADS,
                                pool=pool)
    pool.close()
    pool.join()
