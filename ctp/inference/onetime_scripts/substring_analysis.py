import numpy as np
import pickle

with open('subtrees_WN_subtrees_par_non_par_pairs_parent_1_mean_draw_networks_False_substring_addition_0.p', 'rb') as f:
    subtrees = pickle.load(f)

with open('subtrees_WN_multiple_choice_1.p', 'rb') as f:
    subtrees = pickle.load(f)


def check_if_tree(G):
    num_predecessors = np.array([len(list(G.predecessors(node))) for node in G.nodes()])
    assert(len(np.where(num_predecessors == 0)[0]) == 1)
    assert(np.max(num_predecessors) == 1)
    assert(np.min(num_predecessors) == 0)

def analyze_subtree(gold, predicted):
    assert(gold.nodes() == predicted.nodes())
    terms = gold.nodes()
    substring_pairs = []
    gold_substring_pairs = []
    predicted_substring_pairs = []
    for term_a in terms:
        for term_b in terms:
            if f'_{term_a}' in term_b:
                substring_pairs.append((root, term_a, term_b))
                if (term_a, term_b) in gold.edges():
                    gold_substring_pairs.append((root, term_a, term_b))
                if (term_a, term_b) in predicted.edges():
                    predicted_substring_pairs.append((root, term_a, term_b))
    return substring_pairs, gold_substring_pairs, predicted_substring_pairs


for root, subtrees_dict in subtrees.items():
    check_if_tree(subtrees_dict['predicted'])


possible_substring_trees = []
gold_different_from_substrings = []
predicted_different_from_substrings = []
predicted_different_from_gold = []

substring_edges = []
gold_edges = []
predicted_edges = []
substring_edges_not_gold = []
substring_edges_not_predicted = []
gold_edges_not_predicted = []
predicted_edges_not_gold = []
for root, subtrees_dict in subtrees.items():
    substring_pairs, gold_substring_pairs, predicted_substring_pairs = analyze_subtree(subtrees_dict['gold'], subtrees_dict['predicted'])
    if substring_pairs != gold_substring_pairs:
        gold_different_from_substrings.append(root)
    if gold_substring_pairs != predicted_substring_pairs:
        predicted_different_from_gold.append(root)
    if predicted_substring_pairs != substring_pairs:
        predicted_different_from_substrings.append(root)
    if len(substring_pairs) > 0:
        possible_substring_trees.append(root)

    substring_edges += substring_pairs
    gold_edges += gold_substring_pairs
    predicted_edges += predicted_substring_pairs

    substring_edges_not_gold += list(set(substring_pairs) - set(gold_substring_pairs))
    substring_edges_not_predicted += list(set(substring_pairs) - set(predicted_substring_pairs))
    gold_edges_not_predicted += list(set(gold_substring_pairs) - set(predicted_substring_pairs))
    predicted_edges_not_gold += list(set(predicted_substring_pairs) - set(gold_substring_pairs))
