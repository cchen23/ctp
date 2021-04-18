import json
import networkx as nx
import numpy as np

from ctp.metrics import compute_edge_metrics
from ctp.inference.build_tree_utils import build_tree_MST_CLE
from ctp.inference.examine_subtrees import convert_to_ancestor_graph
from ctp.utils import get_wordnet_df


def compute_random_baseline_subtree_metrics(wordnet_df):
    def build_graph(df):
        '''Build a graph given the relations in df (a pandas dataframe).'''
        G = nx.DiGraph()
        for idx, row in df.iterrows():
            G.add_edge(row["hypernym"], row["term"])
        return G
    metrics_dict = {'p': [], 'r': [], 'f': []}
    for tree_id in np.unique(wordnet_df.tree_id):
        tree_id_df_subset = wordnet_df[wordnet_df["tree_id"] == tree_id]  # Get the rows of the wordnet_df containing pairs for this subtree.
        nodes = list(set(tree_id_df_subset["term"]).union(tree_id_df_subset["hypernym"]))
        gold_tree = build_graph(tree_id_df_subset)
        predicted_graph = nx.DiGraph()
        # Randomly generate the predicted graph.
        nodes = gold_tree.nodes()
        for term in nodes:
            for hypernym in nodes:
                if term != hypernym:
                    weight = np.random.randn()
                    predicted_graph.add_edge(hypernym, term, weight=weight)
        predicted_tree = build_tree_MST_CLE(predicted_graph)
        gold_tree = convert_to_ancestor_graph(gold_tree)
        predicted_tree = convert_to_ancestor_graph(predicted_tree)
        precision, recall, f1 = compute_edge_metrics(predicted_edges=list(predicted_tree.edges()),
            gold_edges=list(gold_tree.edges()))
        metrics_dict['p'].append(precision)
        metrics_dict['r'].append(recall)
        metrics_dict['f'].append(f1)
    return metrics_dict


if __name__ == '__main__':
    languages = ['cat', 'cmn', 'fin', 'fra', 'ita', 'nld', 'pol', 'por', 'spa', 'eng']
    random_seeds = [0, 1, 2]
    all_metrics_dict = {language: {} for language in languages}
    for language in languages:
        print(language)
        if language == 'eng':
            wordnet_filepath = '../datasets/data_creators/df_csvs/bansal14_trees.csv'
        else:
            wordnet_filepath = f'../datasets/data_creators/df_csvs/bansal14_trees_{language}_cleaned.csv'
        wordnet_df = get_wordnet_df(wordnet_filepath, ['hypernym', 'term', 'tree_id', 'train_test_split'], {'header': 0})
        wordnet_df = wordnet_df[wordnet_df.train_test_split == 'test']
        for seed in random_seeds:
            print(seed)
            np.random.seed(seed)
            metrics_dict = compute_random_baseline_subtree_metrics(wordnet_df)
            all_metrics_dict[language][seed] = metrics_dict
    with open('outputs/random_baseline_metrics.json', 'w') as f:
        json.dump(all_metrics_dict, f)
