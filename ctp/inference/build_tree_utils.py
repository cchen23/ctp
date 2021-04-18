import copy
import networkx as nx
import numpy as np

from ctp.inference.networkx_mst import kruskal_mst_edges_anc, kruskal_mst_edges_anc_directed


def build_tree_from_bottom(G, threshold=0.5):
    def get_node_score(G, node):
        score = 0
        for anc, desc, data in G.edges(data=True):
            if anc == node and desc != node:
                score += data["weight"]
        return score

    tree = nx.DiGraph()
    tree.add_nodes_from(G.nodes())
    np.random.seed(
        2020
    )  # Make sure we're not accidentally getting info from node order in nodes list.
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    num_node_descendants = np.array([get_node_score(G, node) for node in nodes])
    node_descendants_order = [nodes[i] for i in np.argsort(num_node_descendants)][::-1]

    for node in node_descendants_order:
        node_predecessors = list(G.predecessors(node))
        node_predecessors = [
            node_predecessor
            for node_predecessor in node_predecessors
            if G[node_predecessor][node]["weight"] > threshold
        ]
        node_descendants = list(nx.descendants(tree, node))
        node_predecessor_descendants_order_indices = [
            node_descendants_order.index(node_predecessor)
            for node_predecessor in node_predecessors
        ]
        for predecessor_index in np.argsort(node_predecessor_descendants_order_indices):
            potential_parent = node_predecessors[predecessor_index]
            if potential_parent not in node_descendants:
                tree.add_edge(potential_parent, node)
                break
    for node in nodes:
        if len(nx.ancestors(tree, node)) == 0:
            weights = [G[par][node]["weight"] if par != node else 0 for par in nodes]
            tree.add_edge(nodes[np.argsort(weights)[-1]], node)
    return tree


def build_tree_MST(G):
    G_undirected = nx.Graph()
    num_bidirectional = 0
    predicted_edges_unweighted = [(edge[0], edge[1]) for edge in G.edges()]
    edge_directions = []  # (hypernym, term) tuples of edge directions.
    for hypernym, term in G.edges():
        weight = G[hypernym][term]["weight"]
        if (term, hypernym) in predicted_edges_unweighted:
            num_bidirectional += 1
            predicted_edges_unweighted.remove((term, hypernym))
            reverse_weight = G[term][hypernym]["weight"]
            if reverse_weight > weight:
                edge_directions.append((term, hypernym))
                G_undirected.add_edge(term, hypernym, weight=reverse_weight)
            else:
                edge_directions.append((hypernym, term))
                G_undirected.add_edge(hypernym, term, weight=weight)
        else:
            edge_directions.append((hypernym, term))
            G_undirected.add_edge(hypernym, term, weight=weight)

    tree_undirected = nx.maximum_spanning_tree(G_undirected)
    tree = nx.DiGraph()
    for (word1, word2) in tree_undirected.edges():
        if (word1, word2) in edge_directions:
            tree.add_edge(word1, word2)
        elif (word2, word1) in edge_directions:
            tree.add_edge(word2, word1)
        else:
            raise Exception(
                "either (word1, word2) or (word2, word1) should be in directions list."
            )
    # print(f'num bidirectional pairs: {num_bidirectional}')
    return tree


def build_tree_MST_CLE(G):
    alg = nx.algorithms.tree.Edmonds(G)
    return alg.find_optimum(style="arborescence")


def build_tree_MST_anc(G):
    G_undirected = nx.Graph()
    num_bidirectional = 0
    predicted_edges_unweighted = [(edge[0], edge[1]) for edge in G.edges()]
    edge_directions = []  # (hypernym, term) tuples of edge directions.
    for hypernym, term in G.edges():
        weight = G[hypernym][term]["weight"]
        if (term, hypernym) in predicted_edges_unweighted:
            num_bidirectional += 1
            predicted_edges_unweighted.remove((term, hypernym))
            reverse_weight = G[term][hypernym]["weight"]
            if reverse_weight > weight:
                edge_directions.append((term, hypernym))
                G_undirected.add_edge(term, hypernym, weight=reverse_weight)
            else:
                edge_directions.append((hypernym, term))
                G_undirected.add_edge(hypernym, term, weight=weight)
        else:
            edge_directions.append((hypernym, term))
            G_undirected.add_edge(hypernym, term, weight=weight)
    tree_edges = kruskal_mst_edges_anc(G_undirected, minimum=False)
    tree = nx.DiGraph()
    for (word1, word2) in tree_edges:
        if (word1, word2) in edge_directions:
            tree.add_edge(word1, word2)
        elif (word2, word1) in edge_directions:
            tree.add_edge(word2, word1)
        else:
            raise Exception(
                "either (word1, word2) or (word2, word1) should be in directions list."
            )
    # print(f'num bidirectional pairs: {num_bidirectional}')
    return tree


def build_tree_MST_anc_directed(G):
    tree_edges = kruskal_mst_edges_anc_directed(G, minimum=False)
    tree = nx.DiGraph()
    for (word1, word2) in tree_edges:
        tree.add_edge(word1, word2)
    # print(f'num bidirectional pairs: {num_bidirectional}')
    return tree


def build_tree_fertility(G, threshold=0.5):
    subtree_IDs = {node: node for node in G.nodes()}
    np.random.seed(
        2020
    )  # Make sure we're not accidentally getting info from node order in nodes list.
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    num_node_descendants = np.array([len(nx.descendants(G, node)) for node in nodes])
    node_descendants_ordered = [nodes[i] for i in np.argsort(num_node_descendants)][
        ::-1
    ]

    tree = nx.DiGraph()

    def update_tree(root):
        descendants = list(nx.descendants(G, root))
        descendants_num_descendants = np.array(
            [
                len(nx.descendants(G, descendant))
                for descendant in descendants
                if G[root][descendant]["weight"] > threshold
            ]
        )
        descendants_ordered = [
            descendants[i] for i in np.argsort(descendants_num_descendants)
        ][::-1]
        for descendant in descendants_ordered:
            if subtree_IDs[descendant] != subtree_IDs[root]:
                descendant_ID = subtree_IDs[descendant]
                root_ID = subtree_IDs[root]
                for node in nodes:
                    if subtree_IDs[node] == descendant_ID:
                        subtree_IDs[node] = root_ID
                tree.add_edge(root, descendant)
                update_tree(descendant)
        return

    update_tree(node_descendants_ordered[0])
    return tree


def build_tree_MST_reorganize(G, initial_method, threshold=0.5, epsilon=1e-4):
    if initial_method == "MST_dir":
        tree = build_tree_MST_anc_directed(G)
    elif initial_method == "MST":
        tree = build_tree_MST(G)
    elif initial_method == "MST_CLE":
        tree = build_tree_MST_CLE(G)

    def reorganize(tree, G):
        G_copy = copy.deepcopy(G)
        for u, v in tree.edges():
            weight_u_v = G[u][v]["weight"]
            v_siblings = list(tree.successors(u))
            for sibling in v_siblings:
                if child == v:
                    continue
                weight_v_sibling = G[v][sibling]["weight"]
                if weight_a_b > threshold and weight_u_v > threshold:
                    G_copy[u][v] = G_copy[u][v] + weight_v_sibling / 10
                    G_copy[v][sibling] = G_copy[u][sibling] + weight_v_sibling / 100
        return G_copy

    G_copy = reorganize(tree, G)
    tree = build_tree_MST_CLE(G_copy)
    # print(f'num bidirectional pairs: {num_bidirectional}')
    return tree
