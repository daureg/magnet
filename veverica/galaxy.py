#! /usr/bin/env python
# vim: set fileencoding=utf-8
"""Initial implementation of galaxy maker"""
# import numpy as np
# import heapq
from collections import namedtuple
from itertools import product
import convert_experiment as cexp
# import cc_pivot as cc
import redensify
from timeit import default_timer as clock
# import graph_tool as gt
Star = namedtuple('Star', 'center points'.split())
vertex_style = {'size': 16, 'font_size': 10}
v_id = 0
spos = None


def profile(func):
    return func


@profile
def sort_degree(G):
    """Return a heap of nodes sorted by degree and a boolean (all false)
    array"""
    res = sorted(G.keys(), key=lambda n: len(G[n]), reverse=True)
    # res = []
    # used = np.zeros(len(G), dtype=np.bool)
    used = [False for _ in G.keys()]
    # for node, adj in G.items():
    #     heapq.heappush(res, (-len(adj), node))
    return res, used


@profile
def edges_of_star(s):
    """Return a list of edges making up a star `s`"""
    return [(s.center, _) if s.center < _ else (_, s.center)
            for _ in s.points]


@profile
def stars_maker(G):
    """Decompose G into a sequence of disjoint stars of decreasing size"""
    degrees, used = sort_degree(G)
    res = []
    redge = []
    # while degrees:
    for center in degrees:
        # center = heapq.heappop(degrees)[1]
        if used[center]:
            continue
        used[center] = True
        star = Star(center, [_ for _ in G[center] if not used[_]])
        # TODO? would be faster to loop over a list
        for p in star.points:
            used[p] = True
        # used[list(star.points)] = True
        res.append(star)
        redge.append(edges_of_star(star))
    return res, redge


@profile
def edge_between(G, s1, s2):
    """return a "arbitrary" edge of `G` between two stars"""
    for u, v in product([s1.center] + s1.points,
                        [s2.center] + s2.points):
        u, v = (u, v) if u < v else (v, u)
        if v in G[u]:
            return u, v
    return None


@profile
def collapse_stars(G, stars):
    """From a graph `G` and its list of `stars`, return a new graph with a
    node for each star a link between them if they are connected in the
    original graph.
    n_mapping: star index in G' -> star center index in G
    e_mapping: edge in G' -> edge in G
    """
    Gprime = {_: set() for _ in range(len(stars))}
    # TODO it turns out that n_mapping is used only for visualization. On the
    # other it doesn't waste so much time
    n_mapping = {}
    # at the end, each node of G will be key, whose value is the star it
    # belongs to
    star_membership = {}
    e_mapping = {}
    for i, s in enumerate(stars):
        n_mapping[i] = s.center
        star_membership[s.center] = i
        for p in s.points:
            star_membership[p] = i
        for j, t in enumerate(stars[i+1:]):
            between = edge_between(G, s, t)
            if between:
                Gprime[i].add(j+i+1)
                Gprime[j+i+1].add(i)
                e_mapping[(i, i+1+j)] = between
    return Gprime, n_mapping, e_mapping, star_membership


@profile
def galaxy_maker_clean(G, k, outname=None):
    """same as galaxy_maker with no visualization and unnecessary variables"""
    current_graph, ems, all_stars = G, [], []
    star_membership = {}
    for i in range(k):
        start = clock()
        stars, stars_edges = stars_maker(current_graph)
        all_stars.append(stars_edges)
        collapsed_graph, _, em, sm = collapse_stars(current_graph, stars)
        if i == 0:
            star_membership = sm
            a, b = set(G.keys()), set(star_membership.keys())
            assert a == b, a - b
        else:
            for orig_nodes, previous_star in star_membership.items():
                star_membership[orig_nodes] = sm[previous_star]
        duration = clock() - start
        print('iteration {} in {:.3f} seconds'.format(str(i).ljust(3),
                                                      duration))
        ems.append(em)
        if outname:
            filename = '{}_{}'.format(outname, i)
            export_spanner(all_stars, ems, star_membership, filename)
        if len(em) == 0 or len(collapsed_graph) == len(current_graph):
            break
        current_graph = collapsed_graph
    return None, None, ems, all_stars, star_membership


@profile
def extract_tree_edges(stars_edges, interstellar_edges):
    res = []
    something_to_remove = False
    if len(interstellar_edges[-1]) != 0:
        # In this case, galaxy_maker didn't collapse the graph all the way down
        # to one node per connected component. Thus we need an extra step.
        # Namely we add all edges betweens stars as part the spanning result
        # graph (instead of taking only those that will the next step graph). A
        # better way would be to extract a spanning tree (TODO ask Fabio about
        # it, because it might also loose the trade off property that stopping
        # early returns more edges but with a lower stretch)
        stars_edges.append([list(interstellar_edges[-1].keys())])
        interstellar_edges.append([])
        something_to_remove = True
    data = zip(reversed(stars_edges), reversed(interstellar_edges[:-1]))
    for current_level_edges, translation_to_lower_level in data:
        res.extend((e for one_star_edges in current_level_edges
                    for e in one_star_edges))
        for i, e in enumerate(res):
            res[i] = translation_to_lower_level[e]
    if something_to_remove:
        stars_edges.pop()
        interstellar_edges.pop()
    return res+[e for star in stars_edges[0] for e in star]


def export_spanner(stars_edges, interstellar_edges, star_membership,
                   filename):
    """Write the active set of edges defined by the arguments in `filename`"""
    size_of_top_graph = len(interstellar_edges[-1])
    final = extract_tree_edges(stars_edges, interstellar_edges)
    with open(filename+'.edges', 'w') as f:
        f.write('\n'.join(('{}, {}'.format(*e) for e in final)))
    with open(filename+'.sm', 'w') as f:
        f.write('\n'.join(('{}\t{}'.format(k, v)
                           for k, v in star_membership.items())))
    edges_mapping = zip(interstellar_edges[-1], final[:size_of_top_graph])
    low_level_edges = [(top_edge, orig_edge)
                       for top_edge, orig_edge in edges_mapping]
    with open(filename+'.lle', 'w') as f:
        f.write('\n'.join(('{}, {}\t{}, {}'.format(*(e0+ek))
                           for e0, ek in low_level_edges)))


def compute_tree_stretch(graph, tree_maps):
    """Compute the distance between any pair of points in the original graph
    and in its restriction by all the tree in tree_maps"""
    from scipy.spatial.distance import squareform
    import graph_tool as gt
    import numpy as np
    graph.set_edge_filter(None)
    E, V = graph.num_edges(), graph.num_vertices()
    dense = E*np.log(V) > V*V
    orig_dst = gt.topology.shortest_distance(graph, dense=dense)
    tree_dsts = []
    for tree_map in tree_maps:
        graph.set_edge_filter(tree_map)
        tree_dsts.append(gt.topology.shortest_distance(graph, dense=dense))
    graph.set_edge_filter(None)

    def distance_vector(dst):
        """transform graph_tool shortest_distance results into a vector of all
        pairwise distances"""
        return squareform(dst.get_2d_array(np.arange(V)))
    return distance_vector(orig_dst), [distance_vector(_) for _ in tree_dsts]


def galaxy_maker(G, k, vizu=False, p=None):
    """Compose collapse and star_maker k times on G"""
    import cc_pivot as cc
    graphs, nms, ems = [G], [], []
    roseta = {_: _ for _ in range(len(G))}
    if vizu:
        kk = cexp.to_graph_tool()
        pos = kk.new_vertex_property('vector<double>')
        pos.set_2d_array(p)
        cc.draw_clustering(kk, pos=pos, vmore=vertex_style)
    all_stars = []
    for i in range(k):
        stars, stars_edges = stars_maker(graphs[-1])
        all_stars.append(stars_edges)
        g, nm, em = collapse_stars(graphs[-1], stars)
        graphs.append(g)
        roseta = {k: roseta[v] for k, v in nm.items()}
        nms.append(nm)
        ems.append(em)
        if vizu:
            print(roseta)
            kk = to_graph_tool_simple(g)
            pos = kk.new_vertex_property('vector<double>')
            pos.set_2d_array(p[:, [roseta[_] for _ in range(len(nm))]])
            name = kk.new_vertex_property('string')
            for n in kk.vertices():
                name[n] = str(roseta[int(n)])
            vertex_style['text'] = name
            cc.draw_clustering(kk, pos=pos, vmore=vertex_style, curved=True)
        if len(em) == 0 or len(g) == len(graphs[-2]):
            break
    return graphs, nms, ems, all_stars


def to_graph_tool_simple(orig):
    """transform a adjacency dict into a graph tool structure, suitable for
    cc_pivot.draw_clustering"""
    import graph_tool as gt
    graph = gt.Graph(directed=False)
    graph.ep['sign'] = graph.new_edge_property('bool')
    graph.vp['cluster'] = graph.new_vertex_property('int')
    graph.add_vertex(len(orig))
    for u, adj in orig.items():
        for v in adj:
            if u < v:
                graph.add_edge(u, v)
    return graph


def _nested_stars(center, branches, level):
    """At level 1, make a star of `branches` nodes plus a `center` at the
    given 2D coordinates. Otherwise do it recursively."""
    import numpy as np
    global v_id, spos
    d = 3**level-1
    centers = np.zeros((2, branches+1))
    centers[:, 0] = center
    sangle = cexp.r.random()*.6
    for i in range(branches):
        centers[:, i+1] = centers[:, 0] + [d*np.cos(i*2*np.pi/branches+sangle),
                                           d*np.sin(i*2*np.pi/branches+sangle)]
    if level == 1:
        if v_id not in redensify.G:
            redensify.G[v_id] = set()
        center_id = v_id
        spos[:, center_id] = centers[:, 0]
        for i in range(branches):
            v_id += 1
            cexp.add_signed_edge(center_id, v_id, True)
            spos[:, v_id] = centers[:, i+1]
        v_id += 1
        return list(range(center_id+1, v_id))
    else:
        res = []
        central_nodes = _nested_stars(centers[:, 0], branches, level-1)
        res.extend(central_nodes)
        for i in range(branches):
            branch_nodes = _nested_stars(centers[:, i+1], branches, level-1)
            for _ in range(level-0):
                cexp.add_signed_edge(cexp.r.choice(central_nodes),
                                     cexp.r.choice(branch_nodes), True)
            res.extend(branch_nodes)
        return res


def _create_nested_star(branches, level):
    """Create a graph using _nested_stars"""
    import numpy as np
    global v_id, spos
    v_id = 0
    spos = np.zeros((2, 3000))
    cexp.new_graph()
    _nested_stars(np.zeros(2), branches, level)
    cexp.finalize_graph()


def _create_star(center_id, center_pos, nb_points):
    """create nb_points around a vertex at center_pos with center_id"""
    import numpy as np
    pos = np.zeros((2, nb_points+1))
    pos[:, 0] = center_pos
    redensify.G[center_id] = set()
    sangle = cexp.r.random()*1.57+.5
    for i in range(nb_points):
        cexp.add_signed_edge(center_id, center_id+i+1, True)
        pos[:, i+1] = pos[:, 0] + [np.cos(i*2*np.pi/nb_points+sangle),
                                   np.sin(i*2*np.pi/nb_points+sangle)]
    return pos


def _make_test_graph():
    """generate a simple star geometry"""
    import numpy as np
    cexp.new_graph()
    p = np.zeros((2, 0))
    C = 2
    for i in range(2):
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+0,
                                                          16*(i//C)+0), 4)])
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+8,
                                                          16*(i//C)+0), 4)])
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+8,
                                                          16*(i//C)+8), 4)])
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+0,
                                                          16*(i//C)+8), 4)])
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+4,
                                                          16*(i//C)+4), 5)])
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+12,
                                                          16*(i//C)+0), 4)])
        p = np.hstack([p, _create_star(len(redensify.G), (16*(i % C)+12,
                                                          16*(i//C)+8), 4)])
        reals = [(36*i+4, 36*i+23), (36*i+21, 36*i+8), (36*i+25, 36*i+12),
                 (36*i+19, 36*i+24), (36*i+11, 36*i+27), (36*i+29, 36*i+32)]
        for u, v in reals:
            cexp.add_signed_edge(u, v, False)
        fakes = [(36*i+2, 36*i+22), (36*i+25, 36*i+9), (36*i+24, 36*i+13),
                 (36*i+16, 36*i+23), (36*i+33, 36*i+28)]
        for u, v in fakes:
            cexp.add_signed_edge(u, v, True)
    cexp.add_signed_edge(35, 39, True)
    cexp.add_signed_edge(9, 66, True)
    cexp.finalize_graph()
    return p


def _full_pipeline(G, pos_array, vizu=True, nb_iter=15):
    from timeit import default_timer as clock
    """Run galaxy on G, show the resulting tree and compute its stretch"""
    import graph_tool as gt
    import cc_pivot as cc
    import numpy as np
    start = clock()
    if vizu:
        _, _, ems, sedge = galaxy_maker(G, nb_iter, vizu, pos_array)
    else:
        _, _, ems, sedge, sm = galaxy_maker_clean(G, nb_iter)
    final = extract_tree_edges(sedge, ems)
    print('Find a tree in {:.3f} seconds'.format(clock()-start))
    kk = cexp.to_graph_tool()
    pos = kk.new_vertex_property('vector<double>')
    pos.set_2d_array(pos_array)
    cls = [[1, 0, 0, .05], [40/255, 100/255, 1, .8]]
    cl = kk.new_edge_property('vector<double>')
    galaxy_tree = kk.new_edge_property('bool')
    if 'text' in vertex_style:
        del vertex_style['text']
    for e in kk.edges():
        u, v = e
        u, v = int(u), int(v)
        cl[e] = cls[(u, v) in final]
        galaxy_tree[e] = (u, v) in final
    vmore = vertex_style
    names = kk.new_vertex_property('string')
    vmore['text'] = names
    cc.draw_clustering(kk, pos=pos, vmore=vmore, emore={'color': cl},
                       osize=1300, curved=True)
    min_span = gt.topology.min_spanning_tree(kk)
    od, tds = compute_tree_stretch(kk, [min_span, galaxy_tree])
    Nlog = np.log(len(G))
    for name, td in zip(['MST', 'Galaxy'], tds):
        print(name)
        print('path stretch / log N at 5, 25, 50 75 and 95 percentile')
        print(np.percentile(td/od, [5, 25, 50, 75, 95])/Nlog)
        print('avg_path_stretch/log N')
        print((td/od).mean()/Nlog)
    return final, tds[1], od


if __name__ == '__main__':
    import real_world as rw
    start = clock()
    rw.read_original_graph('soc-sign-Slashdot090221.txt')
    # rw.read_original_graph('soc-sign-epinions.txt')
    # cexp.generate_random_graph(600, .12)
    # redensify.G, redensify.N, redensify.EDGES_SIGN = cexp.p.load_var('rng22k.my')
    print('Generate graph in {:.3f} seconds'.format(clock()-start))
    redensify.G = rw.G
    redensify.EDGES_SIGN = rw.EDGE_SIGN
    redensify.N = len(rw.G)
    start = clock()
    _, _, ems, sedge, sm = galaxy_maker_clean(redensify.G, 2)
    final = extract_tree_edges(sedge, ems)
    print('Computed tree in {:.3f} seconds'.format(clock()-start))
    with open('tree_slash_early2.dat', 'w') as f:
        f.write('\n'.join(('{}, {}'.format(*e) for e in final)))
