#! /usr/bin/env python
# vim: set fileencoding=utf-8
from glob import glob
from timeit import default_timer as clock

import scratch


def image_size(pbm_filename):
    with open(pbm_filename) as img:
        for i, line in enumerate(img):
            if i == 2:
                w, h = [int(_) for _ in line.strip().split(' ')]
                return w*h


if __name__ == '__main__':
    # pylint: disable=C0103
    num_exp = 4
    files = sorted(glob('belgrade/*.pbm'), key=image_size)
    for f in files:
        G, E = scratch.load_graph(f)
        for noise in [0, 1, 2, 5, 10]:
            Enoisy = scratch.add_noise(E, noise)
            for _ in range(num_exp):
                start = clock()
                name = '{}_{}_{}'.format(f, noise, _)
                scratch.process_graph(G, Enoisy, noise, name, asym=True)
                print('{}{:.1f} secs'.format(name.ljust(60), clock() - start))
