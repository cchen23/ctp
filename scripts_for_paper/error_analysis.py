import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import plotnine as p9

from nltk.corpus import wordnet as wn
from scipy.stats import mannwhitneyu

from ctp.metrics import compute_edge_metrics
from ctp.utils import get_wordnet_df, run_permutation_test

num_permutations = 100000

# Load results and dfs.
results_dir = '../outputs/results'
with open(os.path.join(results_dir, 'subtrees_WN_par_roberta_large_1e6_seed0_ancestor_1_substring_addition_0_MST_CLE.p'), 'rb') as f:
    ancestor_subtrees = pickle.load(f)

with open(os.path.join(results_dir, 'subtrees_WN_par_roberta_large_1e6_seed0_parent_1_substring_addition_0_MST_CLE.p'), 'rb') as f:
    parent_subtrees = pickle.load(f)

data_dir = '../datasets/data_creators/df_csvs/'
synsets_df = get_wordnet_df(os.path.join(data_dir, 'bansal14_trees_synset_names_cleaned_new.csv'), ['hypernym', 'term', 'tree_id', 'train_test_split'], {'header': 0})

wordnet_df = get_wordnet_df(os.path.join(data_dir, 'bansal14_trees_new.csv'), ['hypernym', 'term', 'tree_id', 'train_test_split'])


def wordnet_depth_vs_f1():
    root_depth_list = []
    subtree_f1_list = []
    for subtree_info in ancestor_subtrees.values():
        tree_id = subtree_info['tree_id']
        subtree_df = synsets_df[synsets_df.tree_id == tree_id]
        root = set(subtree_df.hypernym) - set(subtree_df.term)
        root_synset = wn.synset(list(root)[0])
        gold_subtree = subtree_info['gold']
        predicted_subtree = subtree_info['predicted']
        p, r, f1 = compute_edge_metrics(predicted_subtree.edges(), gold_subtree.edges())
        root_depth_list.append(root_synset.max_depth())
        subtree_f1_list.append(f1)

    plt.scatter(root_depth_list, subtree_f1_list)
    plt.xlabel('Root Depth')
    plt.ylabel('Subtree F1')
    plt.title('Depth from WordNet Root vs Subtree F1')
    plt.savefig('./plots/root_depth_vs_subtree_f1.png')
    plt.close()

    pvalue, corr = run_permutation_test(root_depth_list, subtree_f1_list, num_permutations=num_permutations)
    print(f'pvalue of permutation test, correlation between wordnet depth vs f1 correlation {pvalue} (true corr {corr})')


def subtree_depth_vs_recall():
    def get_longest_path(G, source, target):
        return len(max(nx.all_simple_paths(G, source, target), key=lambda x: len(x)))

    edge_info_list = []  # Each item is a (edge_accuracy, descendant_dist_from_root, desc_dist_from_ancestor) tuple.
    for subtree_info in ancestor_subtrees.values():
        tree_id = subtree_info['tree_id']
        subtree_df = wordnet_df[wordnet_df.tree_id == tree_id]
        root = list(set(subtree_df.hypernym) - set(subtree_df.term))[0]
        gold_subtree = subtree_info['gold']
        predicted_subtree = subtree_info['predicted']
        for edge in gold_subtree.edges():
            edge_is_correct = int(edge in predicted_subtree.edges())
            ancestor, descendant = edge[0], edge[1]
            descendant_dist_from_root = get_longest_path(gold_subtree, root, descendant)  # Num nodes inclusive.
            descendant_dist_from_ancestor = get_longest_path(gold_subtree, ancestor, descendant)
            edge_info_list.append((edge_is_correct, descendant_dist_from_root, descendant_dist_from_ancestor))

    edge_correct_list = [edge_info[0] for edge_info in edge_info_list]
    descendant_dist_from_root_list = [edge_info[1] for edge_info in edge_info_list]
    descendant_dist_from_ancestor_list = [edge_info[2] for edge_info in edge_info_list]

    data_df = pd.DataFrame({'edge_correct': edge_correct_list, 'root_dist': descendant_dist_from_root_list, 'anc_dist': descendant_dist_from_ancestor_list})

    # Run permutation tests.
    for root_dist in [3, 4]:
        data_df_subset = data_df[data_df.root_dist == root_dist]
        pvalue_anc, corr_anc = run_permutation_test(list(data_df_subset.edge_correct), list(data_df_subset.anc_dist), num_permutations=num_permutations)
        print(f'pvalue of correlation btwn anc dist conditioned on root_dist {root_dist} and recall pvalue {pvalue_anc}, (true corr {corr_anc})')

    for anc_dist in [2, 3]:
        data_df_subset = data_df[data_df.anc_dist == anc_dist]
        pvalue_root, corr_root = run_permutation_test(list(data_df_subset.edge_correct), list(data_df_subset.root_dist), num_permutations=num_permutations)
        print(f'pvalue of correlation btwn root dist conditioned on anc_dist {anc_dist} and recall pvalue {pvalue_root}, (true corr {corr_root})')

    # Print distance-separated recall values.
    print('Ancestor Edge Recall Table')
    for i in [2, 3, 4]:
        edge_correct_root_dist = data_df[data_df.root_dist == i]['edge_correct']
        percent_correct = np.mean(edge_correct_root_dist)
        print(f'root dist {i}, percent correct {percent_correct}, {len(edge_correct_root_dist)} examples')
    for i in [2, 3, 4]:
        edge_correct_anc_dist = data_df[data_df.anc_dist == i]['edge_correct']
        percent_correct = np.mean(edge_correct_anc_dist)
        print(f'anc dist {i}, percent correct {percent_correct}, {len(edge_correct_anc_dist)} examples')

    for root_dist in [2, 3, 4]:
        for anc_dist in [2, 3, 4]:
            if anc_dist > root_dist:
                continue
            data_df_subset = data_df[(data_df.root_dist == root_dist) & (data_df.anc_dist == anc_dist)]
            edge_correct_root_dist = data_df_subset['edge_correct']
            print(f'root {root_dist}, anc {anc_dist}, recall {"%0.1f" % (np.mean(edge_correct_root_dist) * 100)}')


def category_vs_f1():
    categories = ['abstraction.n.06', 'physical_entity.n.01']

    category_list = []
    subtree_f1_list = []
    roots_list = []
    for subtree_info in ancestor_subtrees.values():
        tree_id = subtree_info['tree_id']
        subtree_df = synsets_df[synsets_df.tree_id == tree_id]
        root = set(subtree_df.hypernym) - set(subtree_df.term)
        root_synset = wn.synset(list(root)[0])
        anc = root_synset
        while(anc.name() not in categories):
            anc = anc.hypernyms()[0]
        gold_subtree = subtree_info['gold']
        predicted_subtree = subtree_info['predicted']
        p, r, f1 = compute_edge_metrics(predicted_subtree.edges(), gold_subtree.edges())
        category_list.append(anc.name())
        subtree_f1_list.append(f1)
        roots_list.append(root_synset.name())

    f1_list_a = [f for c, f in zip(category_list, subtree_f1_list) if c == categories[0]]
    f1_list_b = [f for c, f in zip(category_list, subtree_f1_list) if c == categories[1]]
    print(f'mean {categories[0]} f1: {np.mean(f1_list_a)} ({len(f1_list_a)} trees). mean {categories[1]} f1: {np.mean(f1_list_b)} ({len(f1_list_a)} trees)')
    stat, pvalue = mannwhitneyu(f1_list_a, f1_list_b, alternative='less')
    print(f'stat: {stat}, pvalue: {pvalue}')

    data_df = pd.DataFrame({'Category': [categories[0]] * len(f1_list_a) + [categories[1]] * len(f1_list_b), 'f1': f1_list_a + f1_list_b})
    plot = p9.ggplot(data_df) +\
            p9.geom_histogram(p9.aes(x='f1', y='stat(ndensity)', fill='Category'), binwidth=0.1, alpha=0.3) +\
            p9.labels.labs(x='Subtree F1 Score',
                    title='Distribution of Subtree F1 Scores by Top-Level Category')
    plot.save('./plots/category_f1_hist')


def flatter_trees():
    predicted_longest_paths = []
    gold_longest_paths = []
    num_incorrect_parents = 0
    num_attach_to_ancestor = 0
    for subtree_name, subtree_info in parent_subtrees.items():
        gold_subtree_parent = subtree_info['gold']
        predicted_subtree_parent = subtree_info['predicted']
        for node in gold_subtree_parent.nodes():
            gold_predecessors = list(gold_subtree_parent.predecessors(node))
            predicted_predecessors = list(predicted_subtree_parent.predecessors(node))
            if len(gold_predecessors) > 1 or len(predicted_predecessors) > 1:
                import pdb; pdb.set_trace()
            if predicted_predecessors != gold_predecessors:
                num_incorrect_parents += 1
                if len(predicted_predecessors) > 0 and predicted_predecessors[0] in nx.ancestors(gold_subtree_parent, node):
                    num_attach_to_ancestor += 1
        predicted_longest_paths.append(len(nx.dag_longest_path(predicted_subtree_parent)))
        gold_longest_paths.append(len(nx.dag_longest_path(gold_subtree_parent)))
    print(f'Percentage incorrect attaching to ancestor: {num_attach_to_ancestor}/{num_incorrect_parents}={num_attach_to_ancestor/num_incorrect_parents}')
    print(f'Mean predicted longest path: {np.mean(predicted_longest_paths)}')
    print(f'Mean gold longest path: {np.mean(gold_longest_paths)}')


def percent_correct_root():
    num_correct_roots = 0
    for subtree_name, subtree_info in parent_subtrees.items():
        gold_subtree_parent = subtree_info['gold']
        predicted_subtree_parent = subtree_info['predicted']
        gold_root = [n for n, d in gold_subtree_parent.in_degree() if d == 0]
        predicted_root = [n for n, d in predicted_subtree_parent.in_degree() if d == 0]
        if len(gold_root) > 1 or len(predicted_root) > 1:
            import pdb; pdb.set_trace()
        if gold_root[0] == predicted_root[0]:
            num_correct_roots += 1
    print(f'Percentage correct root: {num_correct_roots}/{len(parent_subtrees)}={num_correct_roots/len(parent_subtrees)}')


if __name__ == '__main__':
    print('***********')
    print('wordnet depth vs f1')
    wordnet_depth_vs_f1()
    print('***********')
    print('subtree depth vs recall')
    subtree_depth_vs_recall()
    print('***********')
    print('category_vs_f1')
    category_vs_f1()
    print('***********')
    print('flatter_trees')
    flatter_trees()
