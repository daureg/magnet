#! /usr/bin/evn python
# vim: set fileencoding=utf-8
"""Read Q and c matrix from a file and write the solution u of
min u.T⋅Q⋅u + c.T⋅u
in another file"""

if __name__ == '__main__':
    # pylint: disable=C0103
    import cvxpy as cvx
    import numpy as np
    from operator import itemgetter
    import sys
    import time
    filename = '__optim_pb_{}.npz'.format(sys.argv[1])
    Q, c = itemgetter('Q', 'c')(np.load(filename))
    u = cvx.Variable(c.size)
    obj = cvx.Minimize(cvx.quad_form(u, Q)+cvx.sum_entries(c.T*u))
    prob = cvx.Problem(obj)
    print('start solving...')
    prob.solve(verbose=True, solver=cvx.SCS, max_iters=4000)
    uval = np.array(u.value).ravel()
    np.savez('__optim_sol_{}.{}'.format(sys.argv[1], int(time.time()),
                                        u=uval))
