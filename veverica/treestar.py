# pylint: disable=W0621
from collections import defaultdict
import random
from copy import deepcopy
from grid_stretch import add_edge
from new_galaxy import extract_stars
from node_classif import short_bfs


def initial_spanning_tree(G, root=None):
    root = random.randint(0, len(G) - 1) if root is None else root
    tree, parents = short_bfs(G, root, [])
    Gbfs = {}
    for u, v in tree:
        add_edge(Gbfs, u, v)
    return Gbfs, parents, tree


def dfs_of_a_tree(G, root, parents, k=2):
    """G must be a tree, that is G[v] doesn't include the parent of v"""
    tree = []
    q = []
    status = [(False, -1) for _ in range(len(G))]
    height = {}
    q.append(root)
    subtrees = []
    subroot = set()
    while q:
        v = q.pop()
        if v >= 0:
            discovered, pred = status[v]
            if v in subroot:
                continue
        else:
            v = -(v + 100)
            discovered, pred = status[v]
            if len(G[v]) == 0:
                height[v] = 0
            else:
                height[v] = 1 + max((height[c] for c in G[v]))
            if height[v] == k or v == root:
                _, stree = short_bfs(G, v, [])
                subtrees.append(stree)
                subroot.add(v)
                p = parents[v]
                if p in G:
                    G[p].remove(v)
        if not discovered:
            status[v] = (True, pred)
            if pred != -1:
                tree.append((v, pred) if v < pred else (pred, v))
            # append the node itself to be poped when all its children are visited
            q.append(-(v + 100))
            for w in sorted(G[v], reverse=True):
                discovered, pred = status[w]
                if pred == -1:
                    status[w] = (discovered, v)
                if not discovered:
                    q.append(w)
    return tree, height, subtrees


def bipartition_edges(all_edges, entities, entities_membership, inner_edges={}):
    """Iterate over `all_edges` and partition them in two groups, those which
    are inside one `entities` (as detected by `entities_membership`) (unless
    there are also in `inner_edges`) and those between two `entities`"""
    within_entities = [set() for _ in entities]
    across_entities = defaultdict(set)
    for u, v in all_edges:
        tu, tv = entities_membership[u], entities_membership[v]
        if tu > tv:
            tu, tv = tv, tu
        if tu == tv and (inner_edges and (u, v) not in inner_edges[tu]):
            within_entities[tu].add((u, v))
        if tu != tv:
            across_entities[(tu, tv)].add((u, v))
    return within_entities, {k: v for k, v in across_entities.items()}

if __name__ == "__main__":
    # pylint: disable=C0103
    from cc_pivot import draw_clustering
    import convert_experiment as cexp
    import draw_utils as du
    from graph_tool import draw as gdraw
    from graph_tool.topology import label_largest_component
    import graph_tool as gt
    import numpy as np
    import seaborn as sns

    N = 50
    is_connected = False
    nb_iter = 0
    while not is_connected and nb_iter < N:
        cexp.fast_random_graph(N, .05)
        g = cexp.to_graph_tool()
        is_connected = label_largest_component(g).a.sum() == N
    pos = gdraw.sfdp_layout(g)

    G, E = cexp.redensify.G, cexp.redensify.EDGES_SIGN
    root = 16
    Gbfs, parents, tree = initial_spanning_tree(G, root)
    ecol, esize = du.color_graph(g, tree)
    tmap = du.map_from_list_of_edges(g, tree)
    g.set_edge_filter(tmap)
    tpos = gdraw.sfdp_layout(g)
    g.set_edge_filter(None)
    ecol, esize = du.color_graph(g, tree)
    g_halo = g.new_vertex_property('bool')
    g_halo.a[root] = True
    draw_clustering(g, pos=tpos, vmore={'size': 22, 'halo': g_halo},
                    emore={'pen_width': esize, 'color': ecol, }, osize=900)

    Bparent = deepcopy(parents)
    Bg = deepcopy(Gbfs)
    for u in Bg:
        p = Bparent[u]
        Bg[u].discard(p)

    dtree, height, strees = dfs_of_a_tree(Bg, root, Bparent)

    cols = sns.color_palette('rainbow', len(strees))
    random.shuffle(cols)
    ecol, esize = du.color_graph(g, tree)
    for e in tree:
        ge = g.edge(*e)
        ecol[ge] = du.light_gray  # [.25,.25,.25,.8]
        esize[ge] = 1.5
    for st, c in zip(strees, cols):
        for u, v in ((u, v) for u, v in st.items() if v is not None):
            ge = g.edge(u, v)
            ecol[ge] = c
            esize[ge] = 3
    draw_clustering(g, pos=tpos, vmore={'size': 22},
                    emore={'pen_width': esize, 'color': ecol}, osize=900)

    tree_membership = {u: i for i, st in enumerate(strees) for u in st}
    tree_root = [[k for k, v in t.items() if v is None][0] for t in strees]
    support_tree = []
    for st in strees:
        support_tree.append({(u, v) if u < v else (v, u)
                             for u, v in st.items() if v is not None})
    within_tree, across_trees = bipartition_edges(E, strees, tree_membership, support_tree)
    Gt = {i: set() for i, u in enumerate(strees)}
    Et = {e: sorted(candidates)[0] for e, candidates in across_trees.items()}
    for e, n in Et.items():
        add_edge(Gt, *e)

    k = gt.Graph(directed=False)
    k.add_vertex(len(strees))
    names = k.new_vertex_property('string')
    vcols = k.new_vertex_property('vector<double>')
    etext = k.new_edge_property('string')
    prev_pos = tpos.get_2d_array((0, 1))
    new_pos = np.zeros((2, len(strees)))
    stpos = k.new_vertex_property('vector<double>')
    for i, (stree_prt, root) in enumerate(zip(strees, tree_root)):
        v = k.vertex(i)
        members = sorted(stree_prt.keys())
        mpos = prev_pos[:, members]
        new_pos[:, i] = mpos.mean(1)
        stpos[v] = prev_pos[:, root]  # mpos.mean(1)
        names[v] = str(root)
        vcols[v] = list(cols[i]) + [.9, ]
    for e, n in Et.items():
        ke = k.add_edge(*e)
        etext[ke] = str(n)
    gdraw.graph_draw(k, stpos, eprops={'text': etext}, output_size=(800, 800),
                     vprops={'pen_width': 0, 'text': names, 'fill_color': vcols, 'size': 24})

    stars, _, star_membership = extract_stars(Gt)
    within_star, across_stars = bipartition_edges(Et, stars, star_membership)

    scols, ssize = k.new_edge_property('vector<double>'), k.new_edge_property('double')
    cols_s = sns.color_palette('Set1', len(stars))
    star_halo = k.new_vertex_property('bool')
    star_halo.a = np.zeros(k.num_vertices())
    for s in stars:
        star_halo[g.vertex(s.center)] = True
    for e in k.edges():
        u, v = int(e.source()), int(e.target())
        su, sv = star_membership[u], star_membership[v]
        if su == sv and stars[su].center in [u, v]:
            scols[e], ssize[e] = list(cols_s[su]) + [.9, ], 3
        else:
            scols[e], ssize[e] = [.2, .2, .2, .8], 1

    stext = k.new_edge_property('string')
    vs_cols = k.new_vertex_property('vector<double>')
    for u in k.vertices():
        vs_cols[u] = list(cols_s[star_membership[int(u)]]) + [.9]
    for candidates in across_stars.values():
        chosen = sorted(candidates)[0]
        ke = k.edge(*chosen)
        ssize[ke] = 3
        stext[ke] = str(Et[chosen])
        scols[ke] = du.black
    gdraw.graph_draw(k, stpos, vprops={'pen_width': 0, 'text': names, 'fill_color': vs_cols, 'size': 24, 'halo': star_halo},
                     eprops={'pen_width': ssize, 'color': scols, 'text': stext}, output_size=(800, 800))
