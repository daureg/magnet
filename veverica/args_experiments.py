# vim: set fileencoding=utf-8
"""Parse experiments command line arguments."""
import argparse
from operator import itemgetter

DATASETS = ['LP', 'ER', 'PA', 'EPI', 'WIK', 'SLA', 'MNIN', 'LR']


def get_parser(desc=None):
    """build a parser with dataset and balance selection"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("data", help="Which data to use",
                        choices=DATASETS, default='PA')
    parser.add_argument("-b", "--balanced", action='store_true',
                        help="Should there be 50/50 +/- edges")
    parser.add_argument("-n", "--noise", help="amount of noise to add",
                        type=int, choices=range(0, 100), default=0)
    parser.add_argument("--short", action='store_true',
                        help="select interstellar edge based on distance")
    parser.add_argument("--safe", action='store_true',
                        help="predict only non ambiguous path")
    parser.add_argument("--seed", default=None, type=int,
                        help="Random number seed")
    return parser


def further_parsing(args):
    data, balanced, noise = itemgetter('data', 'balanced',
                                       'noise')(vars(args))
    seeds = None
    assert noise == 0 or noise >= 1, 'give noise as a percentage'
    synthetic_data = len(data) == 2
    if synthetic_data:
        basename = 'universe/noise'
        if data != 'ER':
            basename += data
        seeds = [100*s + (32 if balanced else 57) for s in range(50)]
    else:
        seeds = list(range(4027, 4047))
        basename = {'EPI': 'soc-sign-epinions.txt',
                    'WIK': 'soc-wiki.txt',
                    'MNI': 'soc-mnist.txt',
                    'MNIN': 'soc-mnist_n.txt',
                    'SLA': 'soc-sign-Slashdot090221.txt'}[data]
        if data.startswith('MNI'):
            seeds = list(range(6000, 6050))
    prefix = basename.split('/')[-1]
    if not synthetic_data:
        prefix = {'WIK': 'wiki', 'MNI': 'mni', 'EPI': 'epi',
                  'SLA': 'slash', 'MNIN': 'mnin'}[data]
        suffixes = ('_bal' if args.balanced else '',
                    '_short' if args.short else '',
                    '_safe' if args.safe else '')
        prefix += '{}{}{}'.format(*suffixes)
    return basename, seeds, synthetic_data, prefix, noise, balanced


def load_raw(basename, redensify, args):
    """Load a graph from a pickled file and shuffle nodes/add noise if
    requested by `args`"""
    import persistent as p
    import convert_experiment as cexp
    import real_world as rw
    redensify.G, redensify.EDGES_SIGN = p.load_var(basename+'.my')
    if args.balanced:
        to_delete = p.load_var(basename+'_delete.my')
        for u, v in to_delete:
            redensify.G[u].remove(v)
            redensify.G[v].remove(u)
            del redensify.EDGES_SIGN[(u, v)]
    seed = args.seed
    if isinstance(seed, int):
        cexp.r.seed(seed)
        rperm = list(redensify.G.keys())
        cexp.r.shuffle(rperm)
        rperm = {i: v for i, v in enumerate(rperm)}
        _ = rw.reindex_nodes(redensify.G, redensify.EDGES_SIGN, rperm)
        redensify.G, redensify.EDGES_SIGN = _
    # Because cexp share the random number generator, given a seed, it
    # will always generate the same noise, which for a one shot case
    # like this makes sense
    if args.noise is not None:
        noise = args.noise/100
        cexp.add_noise(noise, noise)

if __name__ == '__main__':
    args = get_parser('Asym')
    args = args.parse_args()
    print(args)
    print(further_parsing(args))
