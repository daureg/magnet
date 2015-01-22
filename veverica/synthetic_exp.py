# vim: set fileencoding=utf-8
import sys
import os
sys.path.append(os.path.expanduser('~/venvs/34/lib/python3.4/site-packages/'))
import persistent as p
import convert_experiment as cexp
from itertools import repeat
from operator import itemgetter
NUM_THREADS = 14


def process_communities(kwargs):
    _, cluster = cexp.random_signed_communities(4, kwargs['size']//4,
                                                kwargs['size']//10, 0.3, .06,
                                                0.05)
    cexp.redensify.PIVOT_SELECTION = kwargs['pivot']
    delta = sum(cexp.count_disagreements(cluster))
    times, _, errors = cexp.run_one_experiment(100, kwargs['one_at_a_time'])
    return [times, delta, errors]


def run_experiment(pool, process_function, savefile, process_args, n_rep):
    if pool:
        runs = list(pool.imap_unordered(process_function, process_args,
                                        chunksize=n_rep//NUM_THREADS))
    else:
        runs = [process_communities(_) for _ in process_args]
    res = {'time': list(map(itemgetter(0), runs)),
           'delta': list(map(itemgetter(1), runs)),
           'nb_error': list(map(itemgetter(2), runs))}
    p.save_var(savefile, res)


def run_communities_experiment(size, one_at_a_time, pool=None, n_rep=100,
                               pivot=cexp.redensify.PivotSelection.Uniform):
    args = repeat({"size": size, "pivot": pivot,
                   "one_at_a_time": one_at_a_time}, n_rep)
    savefile_name = cexp.savefile_name('communities', [size, 0], pivot,
                                       one_at_a_time)
    run_experiment(pool, process_communities, savefile_name, args, n_rep)


def process_random(kwargs):
    cexp.generate_random_graph(kwargs['size'], pr=0.12)
    cluster = cexp.turn_into_signed_graph_by_propagation(num_cluster=4)
    cexp.add_noise(.06, .06)
    cexp.redensify.PIVOT_SELECTION = kwargs['pivot']
    delta = sum(cexp.count_disagreements(cluster))
    times, _, errors = cexp.run_one_experiment(100, kwargs['one_at_a_time'])
    return [times, delta, errors]


def run_random_experiment(size, one_at_a_time, pool=None, n_rep=100,
                          pivot=cexp.redensify.PivotSelection.Uniform):
    args = repeat({"size": size, "pivot": pivot,
                   "one_at_a_time": one_at_a_time}, n_rep)
    savefile_name = cexp.savefile_name('random', [size, 0], pivot,
                                       one_at_a_time)
    run_experiment(pool, process_random, savefile_name, args, n_rep)


def process_preferential(kwargs):
    cexp.preferential_attachment(kwargs['size'], gamma=1.2)
    cluster = cexp.turn_into_signed_graph_by_propagation(num_cluster=4)
    cexp.add_noise(.06, .06)
    cexp.redensify.PIVOT_SELECTION = kwargs['pivot']
    delta = sum(cexp.count_disagreements(cluster))
    times, _, errors = cexp.run_one_experiment(100, kwargs['one_at_a_time'])
    return [times, delta, errors]


def run_preferential_experiment(size, one_at_a_time, pool=None, n_rep=100,
                                pivot=cexp.redensify.PivotSelection.Uniform):
    args = repeat({"size": size, "pivot": pivot,
                   "one_at_a_time": one_at_a_time}, n_rep)
    savefile_name = cexp.savefile_name('pref', [size, 0], pivot,
                                       one_at_a_time)
    run_experiment(pool, process_preferential, savefile_name, args, n_rep)

if __name__ == '__main__':
    # pylint: disable=C0103
    from multiprocessing import Pool
    cexp.NUM_THREADS = NUM_THREADS
    pool = Pool(NUM_THREADS)
    kind, n = int(sys.argv[1]), int(sys.argv[2])
    exp_per_thread = 5
    strategies = [cexp.redensify.PivotSelection.Uniform,
                  cexp.redensify.PivotSelection.Preferential,
                  cexp.redensify.PivotSelection.ByDegree]
    for s in strategies:
        oaat = s is not strategies[2]
        if kind == 0:
            run_communities_experiment(n, pivot=s, one_at_a_time=oaat,
                                       n_rep=exp_per_thread*NUM_THREADS,
                                       pool=pool)
            if s is strategies[0]:
                run_communities_experiment(n, pivot=s, one_at_a_time=False,
                                           n_rep=exp_per_thread*NUM_THREADS,
                                           pool=pool)
        if kind == 1:
            run_preferential_experiment(n, pivot=s, one_at_a_time=oaat,
                                        n_rep=exp_per_thread*NUM_THREADS,
                                        pool=pool)
            if s is strategies[0]:
                run_preferential_experiment(n, pivot=s, one_at_a_time=False,
                                            n_rep=exp_per_thread*NUM_THREADS,
                                            pool=pool)
        if kind == 2:
            run_random_experiment(n, pivot=s, one_at_a_time=oaat,
                                  n_rep=exp_per_thread*NUM_THREADS,
                                  pool=pool)
            if s is strategies[0]:
                run_random_experiment(n, pivot=s, one_at_a_time=False,
                                      n_rep=exp_per_thread*NUM_THREADS,
                                      pool=pool)
    pool.close()
    pool.join()
