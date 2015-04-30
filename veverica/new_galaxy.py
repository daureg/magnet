#! /usr/bin/env python
# vim: set fileencoding=utf-8
from galaxy import export_spanner
from heap.heap import heap
from collections import namedtuple, defaultdict
from timeit import default_timer as clock
import random
Star = namedtuple('Star', 'center points'.split())


class IdentityDict(dict):
    """Identity dictionary"""

    def __getitem__(self, key):
        """return the key itself as value"""
        return key


def edges_of_star(star):
    center = star.center
    return {(p, center) if p < center else (center, p)
            for p in star.points}


def extract_stars(graph, approx=.0):
    # TODO values could include vertex indice to get stars of same degree in
    # topological order…
    pertub = lambda : 1 + (2*random.random()-1)*approx
    degrees = heap({u: -max(0, int(pertub()*len(adj)))
                    for u, adj in graph.items()})
    # degrees = heap({u: random.randint(0, len(graph)) for u in graph})
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
        used[center] = True
        membership[center] = star_idx
        not_in_stars.remove(center)

        degree_changes = defaultdict(int)
        for p in star.points:
            used[p] = True
            membership[p] = star_idx
            not_in_stars.remove(p)
            for w in graph[p].intersection(not_in_stars):
                degree_changes[w] -= 1
        for node, decrease in degree_changes.items():
            degrees[node] -= decrease
        stars.append(star)
        inner_edges.append(edges_of_star(star))
    assert all(used.values())
    assert len(not_in_stars) == 0
    assert set(membership.keys()) == set(used.keys())
    assert set(membership.values()) == set(range(len(stars)))
    return stars, inner_edges, membership


def collapse_stars(graph, edges, stars, membership, edges_trad, centrality):
    cross_stars_edges = defaultdict(set) if centrality else {}
    new_graph = defaultdict(set)
    # FIXME sorted(edges) will make deterministic if not centrality
    ledges = list(edges)
    random.shuffle(ledges)
    for u, v in ledges:
        star_u, star_v = membership[u], membership[v]
        if star_u > star_v:
            star_u, star_v = star_v, star_u
        if star_u == star_v:
            continue
        if centrality:
            cross_stars_edges[(star_u, star_v)].add((u, v))
        else:
            if (star_u, star_v) not in cross_stars_edges:
                cross_stars_edges[(star_u, star_v)] = (u, v)
                new_graph[star_u].add(star_v)
                new_graph[star_v].add(star_u)

    if centrality:
        def edge_distortion(edge):
            u, v = edges_trad[edge]
            return centrality[u] + centrality[v]

        for (star_u, star_v), candidates in cross_stars_edges.items():
            # candidates = list(candidates)
            # random.shuffle(candidates)
            edge = min(candidates, key=edge_distortion)
            cross_stars_edges[(star_u, star_v)] = edge
            new_graph[star_u].add(star_v)
            new_graph[star_v].add(star_u)

    for star_idx, _ in enumerate(stars):
        if star_idx not in new_graph:
            new_graph[star_idx] = set()
    return new_graph, cross_stars_edges


def galaxy_maker(graph, max_iter, output_name=None, short=False):
    current_graph = graph
    interstellar_edges, stars_edges = [], []
    full_membership, edges_trad = {}, IdentityDict()
    original_node = {u: set([u]) for u in graph}
    centrality = {u: 0 for u in graph} if short else None
    all_inner_edges = []
    for k in range(max_iter):
        start = clock()
        first_iter = k == 0
        edges = {(u, v) for u in current_graph
                 for v in current_graph[u] if u < v}
        stars, inner_edges, star_membership = extract_stars(current_graph)
        all_inner_edges.append(inner_edges)
        if short:
            update_centrality(stars, centrality, original_node)
            long_res = update_nodes_mapping(full_membership, star_membership,
                                            first_iter)
            full_membership, original_node = long_res
        edges -= {e for star_edges in inner_edges for e in star_edges}
        new_graph, outer_edges = collapse_stars(current_graph, edges, stars,
                                                star_membership, edges_trad,
                                                centrality)
        new_trad = {}
        for this_level, prev_level in outer_edges.items():
            new_trad[this_level] = edges_trad[prev_level]
        edges_trad = new_trad
        stars_edges.append(inner_edges)
        interstellar_edges.append(outer_edges)
        current_graph = new_graph
        # print('iteration {}: {:3f}'.format(k+1, clock() - start))
        if first_iter and output_name:
            export_spanner(all_inner_edges, interstellar_edges,
                           full_membership, output_name+'_0')
        if len(outer_edges) == 0:
            break
    final = to_original_edges(stars_edges, interstellar_edges)
    if output_name:
        export_spanner(all_inner_edges, interstellar_edges,
                       full_membership, output_name)
        # with open(output_name, 'w') as output:
        #     output.write('\n'.join(('{}, {}'.format(*e) for e in final)))
    return final, centrality


def update_centrality(stars, centrality, original_node):
    for star in stars:
        for top_p in star.points:
            for real in original_node[top_p]:
                centrality[real] += 1


def update_nodes_mapping(star_membership, new_sm, first_iter):
    """After a collapse step, update the mapping between node indices in the
    new graph and original ones."""
    if first_iter:
        star_membership = new_sm
    else:
        new_mm = {}
        for orig_nodes, previous_star in star_membership.items():
            new_mm[orig_nodes] = new_sm[previous_star]
        star_membership = new_mm
    original_basis = defaultdict(set)
    for orig_id, star_id in star_membership.items():
        original_basis[star_id].add(orig_id)
    return star_membership, original_basis


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
    rw.read_original_graph('soc-wiki.txt')
    # import persistent
    # rw.G, e = persistent.load_var('universe/noiseLP.my')
    start = clock()
    for i in range(1):
        outname = 'fortestonly/wiki_{}.edges'.format(i)
        edges, _ = galaxy_maker(rw.G, 10, output_name=outname, short=True)
    print('{} edges in {:.3f}'.format(len(edges), clock()-start))
