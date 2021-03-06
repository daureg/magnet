#! /usr/bin/python2
# vim: set fileencoding=utf-8
import ujson
import blablacar.spiders.persistent as p
import sys


def initialize():
    p.save_var('seen_users.my', set())
    p.save_var('next_users.my', {'vvUfsO__McfaFcJFSYhaeQ',
                                 'e9OC5d8zznHrePL6ThjyQA',
                                 'cH7eI-InwxyaxIr7_J6GzQ',
                                 'wUEz81t9RmnVdJlWLu32mA',
                                 '7XNi5vaQCDnEpZmW4uhEgQ',
                                 })



def merge():
    seen_users = p.load_var('seen_users.my')
    next_users = p.load_var('next_users.my')
    msg = '{}: {} users are parsed, and {} are to be visited'
    print(msg.format('Before', len(seen_users), len(next_users)))
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
    msg = '{}: {} users are parsed, and {} are to be visited'
    print(msg.format('After', len(seen_users), len(next_users)))
    p.save_var('seen_users.my', seen_users)
    p.save_var('next_users.my', next_users)

if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'init':
        initialize()
    merge()
