#! /usr/bin/python2
# vim: set fileencoding=utf-8
import ujson
import blablacar.spiders.persistent as p
import sys


def initialize():
    p.save_var('seen_users.my', set())
    p.save_var('next_users.my', {'tO9TUCYttaFHA0ixh6_Rcw'})



def merge():
    seen_users = p.load_var('seen_users.my')
    next_users = p.load_var('next_users.my')
    maybe_new = set()
    for filename in sys.argv[1:]:
        if not filename.endswith('.jl'):
            continue
        with open(filename) as f:
            for line in f:
                user = ujson.loads(line)
                uid = user['id']
                seen_users.add(uid)
                next_users.discard(uid)
                maybe_new.update({r['from'] for r in user['reviews']})
    next_users.update(maybe_new - seen_users)
    p.save_var('seen_users.my', seen_users)
    p.save_var('next_users.my', next_users)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'init':
        initialize()
    merge()
