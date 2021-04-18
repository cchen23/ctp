import matplotlib

matplotlib.use("Agg")

import argparse
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
import sys
import copy

sys.path.insert(0, '../')

from metrics import compute_edge_metrics
from networkx.drawing.nx_agraph import graphviz_layout

from ctp.inference.build_tree_utils import build_tree_MST_CLE
from ctp.utils import get_sentences_results_wordnet_df, print_average_metrics, softmax_temp, str2bool


def update_with_oracle(predicted, gold, num_oracle_changes=1):
    '''Updates the tree to put the top num_oracle_changes worst nodes into their correct position.
    args:
        predicted: tree to update.
        gold: gold tree.
        num_oracle_changes: currently not used.
    returns:
        oracle_update_edges_not_tree: List of edges by updating node where update gets best f1.
        oracle_update_edges_tree: List of edges by updating node where update gets best f1, and then running MST.
    '''
    def get_f1_fixed_node(predicted, gold, node):
        predicted_edges = [edge for edge in predicted.edges() if node not in edge] + [edge for edge in gold.edges() if node in edge]
        _, _, f = compute_edge_metrics(predicted_edges=predicted_edges, gold_edges=list(gold.edges()))
        return f

    assert(num_oracle_changes == 1)
    best_f1 = -1
    best_fixed_node = None
    for node in predicted.nodes():
        node_fix_f1 = get_f1_fixed_node(predicted=predicted, gold=gold, node=node)
        if node_fix_f1 > best_f1:
            best_f1 = node_fix_f1
            best_fixed_node = node

    predicted.remove_node(best_fixed_node)
    for edge in gold.edges():
        if best_fixed_node in edge:
            predicted.add_edge(edge[0], edge[1], weight=1e9)

    oracle_update_edges_not_tree = list(predicted.edges())
    oracle_update_edges_tree = list(build_tree_MST_CLE(predicted).edges())
    return oracle_update_edges_not_tree, oracle_update_edges_tree


def draw_trees(gold_tree, predicted_graph, predicted_tree, tree_id):
    '''Draws the gold and predicted tree.
    args:
        gold_tree: The gold tree.
        predicted_graph: The predicted graph.
        predicted_tree: The predicted tree (the predicted tree after MST).
        tree_id: The id of the tree (used in the saved file).
    '''
    precision, recall, f1 = compute_edge_metrics(list(predicted_graph.edges()), list(gold_tree.edges()))
    tree_precision, tree_recall, tree_f1 = compute_edge_metrics(list(predicted_tree.edges()), list(gold_tree.edges()))
    try:
        root = list(nx.topological_sort(gold_tree))[0]
    except Exception:
        root = f"CYCLE_{tree_id}"

    def draw_tree(G, G_invisible, save_name):
        '''Draws the graph G.
        Position according to G_invisible (not sure if this is why we used G_invisible?).
        '''
        nx.draw(G_invisible, pos, alpha=0)
        nx.draw_networkx_labels(G, pos, font_size=5, rotation=45)
        nx.draw(G, pos, node_color="#ffffff")
        plt.savefig(os.path.join(experiment_savedir, f"{save_name}"), dpi=400)
        plt.close()

    node_name_mapping = {node: node.replace('_$_', '_') for node in gold_tree.nodes()}  # Having $ in the node names causes problems with plotting the graph.
    gold_tree = nx.relabel_nodes(gold_tree, node_name_mapping)
    pos = graphviz_layout(gold_tree, prog="dot")
    predicted_tree = nx.relabel_nodes(predicted_tree, node_name_mapping)
    draw_tree(gold_tree, gold_tree, f"{root}_{tree_id}_gold")
    draw_tree(predicted_tree, gold_tree, f'{root}_{tree_id}_pruned_f{str(round(tree_f1, 2)).replace(".", "_")}')


def get_wordnet_data(wordnet_df, tree_id, sentences, results, substring_addition=0):
    def build_graph(df):
        '''Build a graph given the relations in df (a pandas dataframe).'''
        G = nx.DiGraph()
        for idx, row in df.iterrows():
            G.add_edge(row["hypernym"], row["term"])
        return G

    tree_id_df_subset = wordnet_df[wordnet_df["tree_id"] == tree_id]  # Get the rows of the wordnet_df containing pairs for this subtree.
    nodes = list(set(tree_id_df_subset["term"]).union(tree_id_df_subset["hypernym"]))
    gold_tree = build_graph(tree_id_df_subset)
    predicted_graph = nx.DiGraph()

    # Fill the predicted graph edges according to the network's predictions.
    for term in nodes:
        example_ids_subset = [
            example_id
            for example_id, sentence_info in sentences.items()
            if sentence_info["term"] == ' '.join(term.split('_$_'))
        ]
        for hypernym in nodes:
            if term != hypernym:
                pair_example_ids = [
                    example_id
                    for example_id in example_ids_subset
                    if (sentences[example_id]["hypernym"] == ' '.join(hypernym.split('_$_')))
                ]
                # Get the logits corresponding to the network's predictions for this pair (there is one value per pattern).
                pair_logit_values_softmax = []
                for pair_example_id in pair_example_ids:
                    example_prediction = results[pair_example_id]
                    logits = [float(logit) for logit in example_prediction["logits"]]
                    pattern_logit_values = softmax_temp(np.array(logits))[1]
                    # Add the substring addition if the hypernym is a subword of the term.
                    if ('_' + hypernym.lower() in term.lower()):
                        pattern_logit_values += substring_addition
                    pair_logit_values_softmax.append(pattern_logit_values)
                weight = np.mean(pair_logit_values_softmax)
                predicted_graph.add_edge(hypernym, term, weight=weight)
    return gold_tree, predicted_graph


def convert_to_ancestor_graph(G):
    '''Converts a (parent) tree to a graph with edges for all ancestor relations in the tree.'''
    G_anc = nx.DiGraph()
    for node in G.nodes():
        for anc in nx.ancestors(G, node):
            G_anc.add_edge(anc, node)
    return G_anc


def get_weighted_average(metrics_list, num_nodes_list):
    '''Prints the average metric across subtrees, weighted by the size of each subtree.
    args:
        metrics_list: A num_subtrees length list of metrics.
        num_nodes_list: A num_subtrees length list of the number of nodes per subtree.
    '''
    print(
        f"{sum(np.array(metrics_list) * np.array(num_nodes_list)) / sum(num_nodes_list):.2f}"
    )


def run_inference_subtree(tree_id, subtrees_dict, sentences, results, subtrees_info_dict, wordnet_df, prediction_metric_type="ancestor", substring_addition=0, draw_networks=False, num_oracle_changes=0):
    '''Computes flattened and structured prediction metrics for network predictions.
    args:
        tree_id: The id of the subtree for which to run inference.
        subtrees_dict: A dictionary containing the gold tree, predicted graph, and predicted tree for each subtree.
        subtrees_info_dict: A dictionary containing the metrics for each subtree.
        wordnet_df: A dataframe containing parent relations.
        prediction_metric_type: "ancestor" or "parent"; type of metric to predict.
        substring_addition: Amount to add to index 1 of logits if the hypernym is a subword of the term.
        draw_networks: If true, saves images of the predicted tree, gold tree, and predicted graph.
        num_oracle_changes: Number of nodes to move according to the oracle.
    '''
    gold_tree, predicted_graph = get_wordnet_data(
        wordnet_df=wordnet_df,
        tree_id=tree_id,
        sentences=sentences,
        results=results,
        substring_addition=substring_addition
    )
    try:
        root_node = list(nx.topological_sort(gold_tree))[0]
    except Exception:
        root_node = tree_id
    predicted_tree = build_tree_MST_CLE(predicted_graph)
    if prediction_metric_type == "ancestor":
        gold_tree_parent = copy.deepcopy(gold_tree)
        predicted_tree_parent = copy.deepcopy(predicted_tree)
        gold_tree = convert_to_ancestor_graph(gold_tree)
        predicted_tree = convert_to_ancestor_graph(predicted_tree)
    precision, recall, f1 = compute_edge_metrics(predicted_edges=list(predicted_tree.edges()),
            gold_edges=list(gold_tree.edges()))

    subtrees_dict[root_node] = {
        "tree_id": tree_id,
        "gold": gold_tree,
        "predicted": predicted_tree,
        "predicted_graph": predicted_graph,
        "gold_parent": gold_tree_parent,
        "predicted_parent": predicted_tree_parent
    }
    subtrees_info_dict[root_node] = {
        "tree_id": tree_id,
        "subtree_size": len(gold_tree.nodes()),
        "predicted_unpruned_size": len(predicted_graph.nodes()),
        "predicted_pruned_size": len(predicted_tree.nodes()),
        "pruned_precision": precision,
        "pruned_recall": recall,
        "pruned_f1": f1,
    }
    if draw_networks:
        draw_trees(gold_tree, predicted_graph, predicted_tree, tree_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", type=str, help='{experiment-name}.json in experiment_configs/ contains filenames and parameters for the experiment.')
    parser.add_argument("--results-dir", type=str, default="../outputs/results", help='Directory in which network logits are saved.')
    parser.add_argument(
        "--sentences-dir",
        type=str,
        default="../datasets/texeval/generated_training_pairs",
        help='Directory in which input pairs are saved.'
    )
    parser.add_argument('--wordnet-dir',
            type=str,
            help='Directory containing wordnet csv files',
            default='../datasets/data_creators/df_csvs')
    parser.add_argument("--prediction-metric-type", type=str, choices=["ancestor", "parent"], default="ancestor", help='Whether to use ancestor or parent F1 predictions (should use ancestor to compare to other work.')
    parser.add_argument("--softmax-temp", type=float, default=1, help='Temperature parameter for softmax function over network prediction logits.')
    parser.add_argument(
        "--num-dev-trees", type=int, default=np.inf, help="Number of subtrees to test. If >= number of possible trees, runs inference on all available subtrees."
    )
    parser.add_argument("--draw-networks",
        type=str2bool,
        default=False,
        help="whether to save plots of gold pruned graphs")
    parser.add_argument("--save-metrics",
        type=str2bool,
        default=False,
        help="whether to save a json file with metrics and tree sizes for all the subtrees")
    parser.add_argument("--epoch-num", type=int, default=5, help="epoch num to examine")
    parser.add_argument("--substring-addition", type=int, default=0, help='Number added to "is pair" prediction if the hypernym is a substring of the term.')
    parser.add_argument("--config-dir", type=str, default="../experiment_configs", help='Directory containing config files.')
    args = parser.parse_args()

    random.seed(2020)
    np.random.seed(2020)

    with open(os.path.join(args.config_dir, args.experiment_name + ".json")) as results_file:
        results_info_config = json.load(results_file)

    results_filename = results_info_config[
        "results_filename"
    ].format(epoch_num=args.epoch_num)
    sentences_filename = results_info_config["test_filenames"]
    wordnet_filename = results_info_config["wordnet_filename"]

    sentences, results, wordnet_df = get_sentences_results_wordnet_df(
            wordnet_filepath=os.path.join(args.wordnet_dir, wordnet_filename),
            results_filepath=os.path.join(args.results_dir, results_filename),
            sentences_filepath=os.path.join(args.sentences_dir, sentences_filename))

    sentence_keys = np.array(list(sentences.keys()))
    tree_ids = np.unique([val["tree_id"] for val in sentences.values()])

    subtrees_info_dict = {}
    subtrees_dict = {}
    experiment_savedir = os.path.join(args.results_dir,
        f"{args.experiment_name}_{args.prediction_metric_type}_substring_addition_{args.substring_addition}/")
    if not os.path.exists(experiment_savedir):
        os.makedirs(experiment_savedir)

    # Run inference for each subtree.
    for tree_id in tree_ids:
        run_inference_subtree(
            tree_id=tree_id,
            prediction_metric_type=args.prediction_metric_type,
            sentences=sentences,
            results=results,
            draw_networks=args.draw_networks,
            subtrees_dict=subtrees_dict,
            subtrees_info_dict=subtrees_info_dict,
            wordnet_df=wordnet_df,
        )

    # Save results.
    with open(
        os.path.join(args.results_dir,
            f"subtrees_{args.experiment_name}_{args.prediction_metric_type}_{args.softmax_temp}_substring_addition_{args.substring_addition}.p"),
        "wb",
    ) as f:
        pickle.dump(subtrees_dict, f)

    if args.save_metrics:
        with open(
            os.path.join(
                args.results_dir,
                f"subtrees_metrics_{args.experiment_name}_{args.prediction_metric_type}_{args.softmax_temp}_substring_addition_{args.substring_addition}.json",
            ),
            "w",
        ) as f:
            json.dump(subtrees_info_dict, f)

    # Print average metrics over subtrees.
    print_average_metrics(subtrees_info_dict, args.epoch_num)
