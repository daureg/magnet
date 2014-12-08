#! /usr/bin/python2
# vim: set fileencoding=utf-8
"""Run code a lot of time."""


def profile_cc_pivot(n=20, k=100):
    """Run cc_pivot `k` time on a `n` circle."""
    import cc_pivot as cc
    import experiments as xp
    import densify
    orig = xp.make_circle(n)
    densify.random_completion(orig, 0.5)
    for _ in range(k):
        g = orig.copy()
        cc.cc_pivot(g)


def profile_densify(n=40, k=50):
    """Run complete_graph `k` time on a `n` circle."""
    import experiments as xp
    import densify
    for _ in range(k):
        g = xp.make_circle(n)
        densify.complete_graph(g, one_at_a_time=False)


if __name__ == '__main__':
    # pylint: disable=C0103
    profile_densify()
