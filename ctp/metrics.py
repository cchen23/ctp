import networkx as nx
import numpy as np

from typing import Dict


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def compute_scores(results_dict) -> Dict[str, float]:
    results = results_dict.values()
    labels = np.array([result["label"][0] for result in results])
    logits = np.concatenate([np.expand_dims(np.array(result["logits"]), axis=0) for result in results], axis=0)
    label_predictions = np.argmax(logits, axis=1)
    metrics = compute_metrics(labels=labels, label_predictions=label_predictions)
    return metrics


def compute_metrics(labels, label_predictions):
    num_examples = len(labels)
    true_positives = np.count_nonzero((labels == 1) & (label_predictions == 1))
    false_positives = np.count_nonzero((labels == 0) & (label_predictions == 1))
    false_negatives = np.count_nonzero((labels == 1) & (label_predictions == 0))
    true_negatives = np.count_nonzero((labels == 0) & (label_predictions == 0))
    if (
        (true_positives + false_positives == 0)
        or (true_positives + false_negatives == 0)
        or (true_positives == 0)
    ):
        recall = np.nan
        precision = np.nan
        F1 = np.nan
    else:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (true_positives + true_negatives) / num_examples
    metrics = {
        "precision": precision,
        "recall": recall,
        "F1": F1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "accuracy": accuracy,
    }
    metrics_to_print = {k: round(v, 3) for k, v in metrics.items()}
    print(metrics_to_print)
    return metrics_to_print


def compute_edge_metrics(predicted_edges, gold_edges):
    tp = len([edge for edge in predicted_edges if edge in gold_edges])
    fp = len([edge for edge in predicted_edges if edge not in gold_edges])
    fn = len([edge for edge in gold_edges if edge not in predicted_edges])
    if tp == 0:
        # print(f' tp {tp}, fp {fp}, fn {fn}')
        return 0, 0, 0
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (recall * precision) / (recall + precision)
    return precision, recall, f1


def score_tree(tree: nx.DiGraph, relation_type: str, relation_scores: Dict, threshold=0.5):
    '''
    relation: string specifying relation to score (sibling, ancestor, parent).
    relation_scores: dictionary of {(node_A, node_B): score.

    Note: scores should be softmax'ed value of a relation existing.
    '''
    def is_relation(tree, node_A, node_B, relation_type):
        if relation_type == 'sibling':
            return list(tree.predecessors(node_A)) == list(tree.predecessors(node_B))
        elif relation_type == 'parent':
            return node_A in list(tree.predecessors(node_B))
        elif relation_type == 'ancestor':
            return node_A in list(nx.ancestors(tree, node_B))

    nodes_list = list(tree.nodes())
    scores_edge = []
    scores_nonedge = []
    for node_A in nodes_list:
        for node_B in nodes_list:
            if node_A == node_B:
                continue
            pair_relation_score = relation_scores[(node_A, node_B)]
            if is_relation(tree, node_A, node_B, relation_type):
                scores_edge.append(pair_relation_score)
            else:
                scores_nonedge.append(pair_relation_score)
    return scores_edge, scores_nonedge
