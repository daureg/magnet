# vim: set fileencoding=utf-8
"""Exact optimization solution to infer whether two users will give the same
ratings to an item on synthetic and MoviesLens dataset."""
# pylint: disable=E1103,F0401
from collections import defaultdict
import numpy as np
import random


def load_movies():
    """load movies dataset"""
    from operator import itemgetter
    data = np.load('sphere/final_movielens.npz')
    items, ratings = itemgetter('movies', 'ratings')(data)
    items = items[:2929, 4:]
    items /= np.sqrt((items ** 2).sum(-1))[..., np.newaxis]
    return items, ratings


def sample_ratings(ratings, fraction=.5):
    """select a `fraction` of the `ratings` and return a list of corresponding
    users and items"""
    chosen = np.argwhere(ratings)
    users, items = list(set(chosen[:, 0])), list(set(chosen[:, 1]))
    users = np.array(sorted(random.sample(users, int(fraction*len(users)))))
    items = np.array(sorted(random.sample(items, int(fraction*len(items)))))
    return users, items


def infer_user_feature(users_idx, items_idx, items, ratings):
    """fit a linear model for each user to predict its ratings and return the
    corresponding weights vector (i.e. features) + extra intercept param"""
    from sklearn import linear_model
    # linreg = linear_model.LinearRegression()
    linreg = linear_model.Ridge(alpha=.1)
    selected_subset = np.ix_(users_idx, items_idx)
    selected_ratings = ratings[selected_subset]
    selected_items = items[items_idx, :]
    users_features = np.zeros((users_idx.size, 1 + items.shape[1]))
    for i, _ in enumerate(users_idx):
        rated_items = np.argwhere(selected_ratings[i, :]).ravel()
        items_features = selected_items[rated_items]
        items_label = selected_ratings[i, rated_items].astype(int)
        linreg.fit(items_features, items_label)
        users_features[i, :] = [linreg.intercept_] + list(linreg.coef_)
        # Sanity check
        # features = np.hstack([np.ones((items_features.shape[0], 1)),
        #                       items_features])
        # pred = np.round(np.dot(users_features[i, :], features.T))
        # print((pred == items_label).sum() / items_label.size)
        # eval = np.abs(pred - items_label) <= 1
        # print((eval).sum() / items_label.size)
    # TODO: what about normalization? Right now users don't have same norm…
    return users_features


def ratings_similarity(sel_ratings, nb_users):
    """Return a matrix where each row give the index of the closest users based
    on l2 distance between all ratings (including zero one)"""
    closer_by_ratings = np.zeros((nb_users, nb_users))
    for i in range(nb_users):
        uratings = sel_ratings[i, :]
        urated = uratings > 0
        approx_dst = (uratings[urated] - sel_ratings[i+1:, urated])**2
        approx_dst = approx_dst.sum(-1)
        closer = np.argsort(approx_dst)
        closer_by_ratings[i, :len(closer)] = closer+i+1
    return closer_by_ratings.astype(int)


def build_graph(sel_items, sel_ratings, nb_users, users_features,
                max_close_users=10, min_common_items=3, max_multi_edges=4):
    """create a list of edges between users with enough common equal ratings"""
    closer_by_ratings = ratings_similarity(sel_ratings, nb_users)
    assert nb_users == sel_ratings.shape[0], "can't remove arg"
    n = nb_users
    edges = []
    adjacency = defaultdict(set)
    for u in range(n):
        uratings = sel_ratings[u, :]
        urated = uratings > 0
        u_vec = users_features[u, :]
        for v in closer_by_ratings[u, :min(max_close_users, n-u+1)]:
            vratings = sel_ratings[v, :]
            common = np.logical_and(urated, vratings > 0)
            same = uratings[common] == vratings[common]
            if same.sum() < min_common_items:
                continue
            X = np.ones((int(same.sum()), 1+sel_items.shape[1]))
            diff_vec = u_vec - users_features[v, :]
            X[:, 1:] = sel_items[common, :][same, :]
            diff_ratings = np.abs(np.dot(diff_vec, X.T))
            same_items_idx = np.argwhere(common).ravel()[same]
            adjacency[u].add(v)
            adjacency[v].add(u)
            for x in np.argsort(diff_ratings)[:max_multi_edges]:
                edges.append((u, v, same_items_idx[x], diff_ratings[x]))
    return edges, adjacency


def greedy_dominating_set(adjacency, nb_users):
    """Return a dominating set of nodes of the graph represented by its
    `adjacency` lists"""
    nodes = set(adjacency.keys())
    node_uncovered = np.ones(nb_users, dtype=np.bool)
    dominating = set()
    while np.any(node_uncovered):
        nodes_to_search = [u for u in nodes if u not in dominating]
        gains = [node_uncovered[list(adjacency[u])].sum()
                 for u in nodes_to_search]
        if sum(gains) == 0:
            break
        v = nodes_to_search[np.argmax(gains)]
        dominating.add(v)
        node_uncovered[v] = False
        node_uncovered[list(adjacency[v])] = False
    return dominating


def hide_some_users(adjacency, nb_users, fraction_of_hidden=.3,
                    visible_seed=None):
    """Set each node to be hidden or not and return the indices of the hidden
    ones"""
    nodes = set(adjacency.keys())
    isolated = set(range(nb_users)) - nodes
    visible_seed = visible_seed or greedy_dominating_set(adjacency, nb_users)
    already_visible = len(visible_seed) + len(isolated)
    fraction_of_hidden *= nb_users / (nb_users - already_visible)
    hidden, visibility = set(), {}
    for u in range(nb_users):
        if u in isolated or u in visible_seed:
            visibility[u] = True
            continue
        if random.random() < fraction_of_hidden:
            hidden.add(u)
            visibility[u] = False
        else:
            visibility[u] = True
    user_order = sorted(hidden)
    return user_order, visibility


def set_up_problem(users_features, items_features, edges, visibility,
                   hidden_user_idx):
    """Given a graph with some hidden user, build the optimization problem to
    recover them"""
    import quad_program as qp
    qp.FEATURE_START = 0
    qp.USERS = users_features
    qp.ITEMS = np.hstack([np.ones((items_features.shape[0], 1)),
                          items_features])
    assert qp.USERS.shape[1] == qp.ITEMS.shape[1]
    qp.DIM_SIZE = users_features.shape[1]
    return qp.construct_optimization_matrix(edges, visibility, hidden_user_idx)


def binarize_ratings(one_user_ratings):
    """Binarize `one_user_ratings` into liked and not liked items in the more
    balanced way possible. Also return which items were rated to distinguis
    between the two kind of zeros (not rated vs rated and not liked)"""
    rated = one_user_ratings > 0
    threshold = np.median(one_user_ratings[rated])
    liked_eq = (one_user_ratings[rated]>=threshold).astype(int)
    liked_strict = (one_user_ratings[rated]>threshold).astype(int)
    eq_num_one, strict_num_one = liked_eq.sum(), liked_strict.sum()
    # find the more balanced way of picking classes
    if abs(eq_num_one - len(liked_eq)/2) < abs(strict_num_one - len(liked_eq)/2):
        return (one_user_ratings>=threshold).astype(int), rated
    return (one_user_ratings>threshold).astype(int), rated

def get_test_edges(sel_ratings, nb_users, max_close_users=10,
                   min_common_items=3):
    """return a list of edges between users along with similarity feedback
    (same binary or integral ratings)"""
    closer_by_ratings = ratings_similarity(sel_ratings, nb_users)
    n = nb_users
    test_edges = []
    bin_test_edges = []
    for u in range(n):
        uratings = sel_ratings[u, :]
        uliked, urated = binarize_ratings(uratings)
        how_far_to_look = min(max_close_users, n-u+1)
        for v in sorted(closer_by_ratings[u, :how_far_to_look]):
            vratings = sel_ratings[v, :]
            vliked, vrated = binarize_ratings(vratings)
            common = np.logical_and(urated, vrated)
            if common.sum() < min_common_items:
                continue
            cm_ratings = uratings[common] == vratings[common]
            cm_liked = uliked[common] == vliked[common]
            cm_index = np.argwhere(common).ravel()
            test_edges.extend(((u, v, item_idx, same_feedback)
                               for item_idx, same_feedback in zip(cm_index,
                                                                  cm_ratings)))
            bin_test_edges.extend(((u, v, item_idx, same_feedback)
                               for item_idx, same_feedback in zip(cm_index,
                                                                  cm_liked)))
    return test_edges, bin_test_edges

if __name__ == '__main__':
    pass
