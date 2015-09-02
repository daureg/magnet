#! /usr/bin/env python
# vim: set fileencoding=utf-8
import scratch
from timeit import default_timer as clock

if __name__ == '__main__':
    # pylint: disable=C0103
    num_exp = 4
    f = 'belgrade/gplus.my'
    osn = ['belgrade/{}.my'.format(_)
           for _ in ['wiki', 'slash', 'epi']]
    for f in osn:
        G, E = scratch.load_graph(f)
    # for noise in [0, 1, 2, 5, 10]:
        noise = 0
        Enoisy = scratch.add_noise(E, noise)
        for _ in range(num_exp):
            start = clock()
            name = '{}_{}_{}'.format(f, noise, _)
            scratch.process_graph(G, Enoisy, noise, name, asym=True)
            print('{}{:.1f} secs'.format(name.ljust(60), clock() - start))
