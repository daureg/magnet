# vim: set fileencoding=utf-8
"""Parse experiments command line arguments."""
import argparse
from operator import itemgetter


def get_parser(desc=None):
    """build a parser with dataset and balance selection"""
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("data", help="Which data to use",
                        choices=['ER', 'PA', 'EPI', 'WIK', 'SLA', 'MNI'],
                        default='PA')
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
    # data = args[1].upper()
    # balanced = bool(int(args[2]))
    # noise = None if len(args) < 4 else float(args[3])
    seeds = None
    assert data in ['ER', 'PA', 'EPI', 'WIK', 'SLA']
    assert noise == 0 or noise >= 1, 'give noise as a percentage'
    synthetic_data = len(data) == 2
    if synthetic_data:
        basename = 'universe/noise'
        if data == 'PA':
            basename += 'PA'
        seeds = [100*s + (32 if balanced else 57) for s in range(50)]
    else:
        seeds = list(range(4027, 4047))
        basename = {'EPI': 'soc-sign-epinions.txt',
                    'WIK': 'soc-wiki.txt',
                    'SLA': 'soc-sign-Slashdot090221.txt'}[data]
    prefix = basename.split('/')[-1]
    if not synthetic_data:
        prefix = {'WIK': 'wiki', 'EPI': 'epi', 'SLA': 'slash'}[data]
        if balanced:
            prefix += '_bal'
    return basename, seeds, synthetic_data, prefix, noise, balanced

if __name__ == '__main__':
    args = get_parser('Asym')
    args = args.parse_args()
    print(args)
    print(further_parsing(args))
