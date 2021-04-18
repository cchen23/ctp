"""
Algorithms for calculating min/max spanning trees/forests.

"""
from heapq import heappop, heappush
from operator import itemgetter
from itertools import count
from math import isnan

import copy
import networkx as nx
import numpy as np
from networkx.utils import UnionFind, not_implemented_for


def kruskal_mst_edges_anc(
    G, minimum, weight="weight", keys=True, data=True, ignore_nan=False
):
    """Iterate over edges of a Kruskal's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    subtrees = UnionFind(elements=G.nodes())
    nodes = list(G.nodes())
    included_edges = []
    G_copy = copy.deepcopy(G)

    def filter_nan_edges(edges, weight=weight):
        for u, v, d in edges:
            wt = d.get(weight, 1)
            yield wt, u, v, d

    def update_weights(G, u, v):
        updated_subtree = subtrees[u]
        for node in nodes:
            if node in [u, v]:
                continue

            u_to_node_weight = G_copy[u][node]["weight"]
            G[v][node]["weight"] = G[v][node]["weight"] + u_to_node_weight
        return G

    def find_next_edge(G):
        iteration_edges = sorted(
            filter_nan_edges(G.edges(data=True)), key=itemgetter(0)
        )[::-1]
        for wt, u, v, d in iteration_edges:
            if subtrees[u] != subtrees[v]:
                G = update_weights(G, u, v)
                included_edges.append((u, v))
                subtrees.union(u, v)
                return G

    while len(np.unique(list(subtrees.parents.values()))) > 1:
        G = find_next_edge(G)
    return included_edges


def kruskal_mst_edges_anc_directed(
    G, minimum, weight="weight", keys=True, data=True, ignore_nan=False
):
    """Iterate over edges of a Kruskal's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    nodes = list(G.nodes())
    subgraphs = {node: nx.DiGraph() for node in nodes}
    subtree_IDs = {node: node for node in nodes}
    included_edges = []
    G_copy = copy.deepcopy(G)

    def filter_nan_edges(edges, weight=weight):
        for u, v, d in edges:
            wt = d.get(weight, 1)
            yield wt, u, v, d

    def update_weights(G, u, v):
        updated_subtree = subgraphs[u]
        v_ancestors = nx.ancestors(updated_subtree, v)
        for node in nodes:
            if node in [u, v]:
                continue
            count = 1
            try:
                anc_weights_sum = G[v][node]["weight"]
            except:
                import pdb

                pdb.set_trace()
            for anc in v_ancestors:
                if anc == node:
                    continue
                count += 1
                try:
                    anc_weights_sum += G[anc][node]["weight"]
                except:
                    import pdb

                    pdb.set_trace()
            try:
                G[v][node]["weight"] = anc_weights_sum / count
            except:
                import pdb

                pdb.set_trace()
        return G

    def find_next_edge(G):
        iteration_edges = sorted(
            filter_nan_edges(G.edges(data=True)), key=itemgetter(0)
        )[::-1]
        for wt, u, v, d in iteration_edges:
            if subtree_IDs[u] != subtree_IDs[v] and subtree_IDs[v] == v:
                v_ID = subtree_IDs[v]
                u_ID = subtree_IDs[u]
                for node in nodes:
                    if subtree_IDs[node] == v_ID:
                        subtree_IDs[node] = u_ID
                subgraphs[u].add_edge(u, v, weight=wt)
                G = update_weights(G, u, v)
                included_edges.append((u, v))
                return G

    while len(np.unique(list(subtree_IDs.values()))) > 1:
        G = find_next_edge(G)
    return included_edges
