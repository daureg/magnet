#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Run experiments in parallel."""
import experiments as xp

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    pool = Pool(10)
    # for n_squares in [65, 82, 100]:
    #     xp.run_ring_experiment(2+2*n_squares, n_squares, n_rep=350, pool=pool)
    # xp.run_circle_experiment(20, pool=pool)
    # xp.run_circle_experiment(20, one_at_a_time=False, pool=pool)
    n_squares = 30
    # xp.run_ring_experiment(2+2*n_squares, n_squares, pool=pool)
    xp.run_ring_experiment(2+2*n_squares, n_squares, one_at_a_time=False,
                           pool=pool)
    # for ball_size, nb_balls in [(28,10)]:
    #     xp.run_planted_experiment(ball_size, nb_balls, n_rep=200, pool=pool)
    #     xp.run_planted_experiment(ball_size, nb_balls, n_rep=200, pool=pool,
    #                               by_degree=True)
    #     xp.run_planted_experiment(ball_size, nb_balls, n_rep=200, pool=pool,
    #                               by_betweenness=True)
    #     xp.run_planted_experiment(ball_size, nb_balls, n_rep=200, pool=pool,
    #                               one_at_a_time=True)
    pool.close()
    pool.join()
