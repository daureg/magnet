#! /usr/bin/env python
# vim: set fileencoding=utf-8
from timeit import default_timer as clock

import grid_stretch as gs
import persistent


def read_img(filename):
    img = []
    row = ''
    with open(filename) as f:
        for i, line in enumerate(f.readlines()):
            if i < 2:
                continue
            if i == 2:
                w, h = [int(_) for _ in line.split(' ')]
                continue
            line = line.strip()
            need_to_add = w - len(row)
            if len(line) < need_to_add:
                row += line
                continue
            row += line[:need_to_add]
            img.append([int(_) for _ in row])
            row = line[need_to_add:]
        assert len(img) == h
    return img, w, h


def build_graph(img, w, h):
    G, E = {}, {}
    node_signs = {}
    for i, row in enumerate(img):
        for j, pxl in enumerate(row):
            neighbors = []
            if i > 0:
                neighbors.append((i-1, j))
            if i < h - 1:
                neighbors.append((i+1, j))
            if j > 0:
                neighbors.append((i, j-1))
            if j < w - 1:
                neighbors.append((i, j+1))
            u = i*w + j
            node_signs[u] = 2*pxl-1
            for iv, jv in neighbors:
                v = iv*w + jv
                gs.add_edge(G, u, v)
                edge = (u, v) if u < v else (v, u)
                E[edge] = pxl == img[iv][jv]
    return node_signs, G, E

if __name__ == '__main__':
    # pylint: disable=C0103
    from glob import glob
    num = lambda f: -int(f.split('_')[-1].split('.')[0])
    files = sorted(glob('*.pbm'), key=num)
    for input_image in files:
        start = clock()
        output_graph = 'belgrade/'+input_image.replace('.pbm', '.my')
        persistent.save_var(output_graph, build_graph(*read_img(input_image)))
        print('done {} in {:.3f}'.format(input_image, clock() - start))
