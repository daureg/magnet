#! /usr/bin/env python
# vim: set fileencoding=utf-8
from glob import glob
from timeit import default_timer as clock

import scratch

if __name__ == '__main__':
    # pylint: disable=C0103
    num_exp = 4
    sizes = [int(10*(2**(i/2))) for i in range(0, 19)]
    for size in sizes:
        f = 'grid_{}'.format(size)
        G, E = scratch.load_graph('grid', size)
        for noise in [0, 1, 2, 5, 10]:
            Enoisy = scratch.add_noise(E, noise)
            for _ in range(num_exp):
                start = clock()
                name = '{}_{}_{}'.format(f, noise, _)
                scratch.process_graph(G, Enoisy, noise, name, asym=True)
                print('{}{:.1f} secs'.format(name.ljust(60), clock() - start))
