#! /usr/bin/python2
# vim: set fileencoding=utf-8
import pickle as pickle


def save_var(filename, d, proto=3):
    with open(filename, 'wb') as f:
        pickle.dump(d, f, proto)


def load_var(filename):
    try:
        with open(filename, 'rb') as f:
            d = pickle.load(f)
    except IOError:
        raise
    return d

def resave(filename, proto=3):
    data = load_var(filename)
    save_var(filename, data, proto)
