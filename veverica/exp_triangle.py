#! /usr/bin/env python
# vim: set fileencoding=utf-8
import scratch
from glob import glob
from timeit import default_timer as clock

if __name__ == '__main__':
    # pylint: disable=C0103
    num_exp = 4
    t_size = lambda x: int(x.split('_')[-1].split('.')[0])
    files = sorted(glob('belgrade/triangle_*.my'), key=t_size)
    for f in files:
        print(f)
        G, E = scratch.load_graph(f)
        for noise in [0, 1, 2, 5, 10]:
            Enoisy = scratch.add_noise(E, noise)
            for _ in range(num_exp):
                start = clock()
                name = '{}_{}_{}'.format(f, noise, _)
                scratch.process_graph(G, Enoisy, noise, name, asym=True)
                print('{}{:.1f} secs'.format(name.ljust(60), clock() - start))
