#! /usr/bin/env python
# vim: set fileencoding=utf-8
from heap.heap import heap
from collections import namedtuple, defaultdict
from timeit import default_timer as clock
Star = namedtuple('Star', 'center points'.split())


def edges_of_star(star):
    center = star.center
    return {(p, center) if p < center else (center, p)
            for p in star.points}


def extract_stars(graph):
    # TODO values could include vertex indice to get stars of same degree in
    # topological order…
    degrees = heap({u: -len(adj) for u, adj in graph.items()})
    used = {u: False for u in graph}
    not_in_stars = set(graph.keys())
    stars, inner_edges = [], []
    membership = {}
    while degrees:
        star_idx = len(stars)
        center = degrees.pop()
        if used[center]:
            continue
        star = Star(center, [p for p in graph[center] if not used[p]])
        membership[center] = star_idx
        not_in_stars.remove(center)
        degree_changes = defaultdict(int)
        for p in star.points:
            used[p] = True
            not_in_stars.remove(p)
            membership[p] = star_idx
            for w in graph[p].intersection(not_in_stars):
                degree_changes[w] -= 1
        for node, decrease in degree_changes.items():
            degrees[node] -= decrease
        stars.append(star)
        inner_edges.append(edges_of_star(star))
    return stars, inner_edges, membership


def collapse_stars(graph, edges, stars, membership):
    cross_stars_edges = {}
    new_graph = defaultdict(set)
    for u, v in sorted(edges):
        star_u, star_v = membership[u], membership[v]
        if star_u > star_v:
            star_u, star_v = star_v, star_u
        if star_u == star_v or (star_u, star_v) in cross_stars_edges:
            continue
        cross_stars_edges[(star_u, star_v)] = (u, v)
        new_graph[star_u].add(star_v)
        new_graph[star_v].add(star_u)
    return new_graph, cross_stars_edges


def galaxy_maker(graph, max_iter, output_name):
    current_graph = graph
    interstellar_edges, stars_edges = [], []
    for k in range(max_iter):
        start = clock()
        edges = {(u, v) for u in current_graph.keys()
                 for v in current_graph[u] if u < v}
        stars, inner_edges, star_membership = extract_stars(current_graph)
        edges -= {e for star_edges in inner_edges for e in star_edges}
        new_graph, outer_edges = collapse_stars(current_graph, edges, stars,
                                                star_membership)
        stars_edges.append(inner_edges)
        interstellar_edges.append(outer_edges)
        current_graph = new_graph
        print('iteration {}: {:3f}'.format(k+1, clock() - start))
        if len(outer_edges) == 0:
            break
    final = to_original_edges(stars_edges, interstellar_edges)
    with open(output_name, 'w') as output:
        output.write('\n'.join(('{}, {}'.format(*e) for e in final)))
    return final


def to_original_edges(stars_edges, interstellar_edges):
    assert interstellar_edges[-1] == {}, "not finished"
    final_edges = []
    data = zip(reversed(stars_edges), reversed(interstellar_edges[:-1]))
    for current_level_edges, translation_to_lower_level in data:
        final_edges.extend((e for one_star_edges in current_level_edges
                            for e in one_star_edges))
        for i, e in enumerate(final_edges):
            final_edges[i] = translation_to_lower_level[e]
    return final_edges+[e for star in stars_edges[0] for e in star]


if __name__ == '__main__':
    # pylint: disable=C0103
    import real_world as rw
    outname = 'fortestonly/wiki'
    rw.read_original_graph('soc-sign-epinions.txt')
    start = clock()
    _ = galaxy_maker(rw.G, 10, outname)
    print('{} edges in {:.3f}'.format(len(_), clock()-start))
