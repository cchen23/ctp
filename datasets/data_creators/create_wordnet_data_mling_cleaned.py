import argparse
import hashlib
import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
from nltk.corpus import wordnet as wn
from wordnet_reconstruction.utils import get_wordnet_df

sys.path.append("..")

np.random.seed(2020)


def write_sentences(df, tree_ids, save_filename, anc_label, sib_label, desc_label, rand_label, parent_label, language, wordnet_df_subset, split):
    """
    df: dataframe containing pair information ('term' and 'hypernym' columns).
    save_filename: filename to save data to.
    """

    def build_graph(df):
        G = nx.DiGraph()
        for idx, row in df.iterrows():
            G.add_edge(row['hypernym'], row['term'])
        return G

    def create_example(pattern, term, hypernym, label, data_dict, tree_id):
        '''
        args:
            if all_anc_desc=True, make examples with all ancestors and descendants. Else randomly choose one of each.
        '''
        term = term.replace('_$_', ' ')
        hypernym = hypernym.replace('_$_', ' ')
        sentence = pattern.replace('[TERM]', term).replace('[HYPERNYM]', hypernym).replace('_', ' ')
        example = {'sentence': sentence, 'label': label, 'tree_id': tree_id, 'term': term, 'hypernym': hypernym}

        example_id = hashlib.sha224(sentence.encode('utf-8')).hexdigest()
        data_dict[example_id] = example
        return

    data_dict = dict()

    for tree_id in tree_ids:
        df_subset = df[df['tree_id'] == tree_id]
        nodes = list(set(df_subset['term']).union(df_subset['hypernym']))
        G = build_graph(df_subset)

        # Remove all nodes without translations.
        root_node = list(nx.topological_sort(G))[0]
        if len(wn.synset(root_node).lemma_names(language)) == 0:
            continue
        missing_nodes = [node for node in nodes if len(wn.synset(node).lemma_names(language)) == 0]
        G.remove_nodes_from(missing_nodes)

        # Remove all nodes that are not connected to the root.
        connected_descendants = list(nx.descendants(G, root_node))
        unconnected_descendants = set(nodes) - set(connected_descendants + [root_node])
        G.remove_nodes_from(unconnected_descendants)

        # Rename nodes to translations.
        node_label_mapping = {node: wn.synset(node).lemma_names(language)[0] for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping=node_label_mapping)
        nodes = list(G.nodes())

        # Update wordnet_df_subset.
        for edge in G.edges():
            wordnet_df_subset = wordnet_df_subset.append({'hypernym': edge[0], 'term': edge[1], 'tree_id': tree_id, 'train_test_split': split}, ignore_index=True)

        wn_synsets = []
        for node in nodes:
            wn_synsets += wn.synsets(node)
        for node1 in nodes:
            predecessors = set(G.predecessors(node1))
            ancestors = list(set(nx.ancestors(G, node1)) - predecessors)
            descendants = list(nx.descendants(G, node1))
            siblings = set()
            for predecessor in predecessors:
                siblings = siblings.union(list(set(list(G.successors(predecessor))) - set([node1])))
            siblings = list(siblings)
            for node2 in nodes:
                pair_patterns = ['[TERM] IS A [HYPERNYM]']
                for pattern in pair_patterns:
                    if node1 == node2:
                        continue
                    if node2 in predecessors:
                        create_example(pattern, node1, node2, parent_label, data_dict, tree_id)
                    elif node2 in ancestors:
                        create_example(pattern, node1, node2, anc_label, data_dict, tree_id)
                    elif node2 in siblings:
                        create_example(pattern, node1, node2, sib_label, data_dict, tree_id)
                    elif node2 in descendants:
                        create_example(pattern, node1, node2, desc_label, data_dict, tree_id)
                    else:
                        create_example(pattern, node1, node2, rand_label, data_dict, tree_id)
    with open(save_filename, 'w') as f:
        json.dump(data_dict, f)

    positive_examples = {k: v for k, v in data_dict.items() if v['label'] == 1}
    negative_examples = {k: v for k, v in data_dict.items() if v['label'] == 0}
    sampled_negative_ids = np.random.choice(list(negative_examples.keys()), size=len(positive_examples.keys()), replace=False)
    negative_examples_subset = {k: data_dict[k] for k in sampled_negative_ids}
    balanced_data_dict = {**positive_examples, **negative_examples_subset}
    with open(save_filename.replace('.json', '_balanced.json'), 'w') as f:
        json.dump(balanced_data_dict, f)

    return wordnet_df_subset

def generate_data(data_dir,
                  save_dir,
                  anc_label,
                  sib_label,
                  desc_label,
                  rand_label,
                  parent_label,
                  language):
    wordnet_filepath = os.path.join(data_dir, 'bansal14_trees_synset_names_cleaned_new.csv')
    wordnet_column_names = ['hypernym', 'term', 'tree_id', 'train_test_split']
    wordnet_df = get_wordnet_df(wordnet_filepath, wordnet_column_names, {'header': 0})
    tree_ids_train = np.unique(wordnet_df[wordnet_df['train_test_split'] == 'train']['tree_id'])
    tree_ids_dev = np.unique(wordnet_df[wordnet_df['train_test_split'] == 'dev']['tree_id'])
    tree_ids_test = np.unique(wordnet_df[wordnet_df['train_test_split'] == 'test']['tree_id'])
    wordnet_df_subset = pd.DataFrame(columns=wordnet_column_names)

    wordnet_df_subset = write_sentences(wordnet_df,
                    tree_ids_train,
                    os.path.join(save_dir, f'wordnet_bansal_anc_{anc_label}_sib_{sib_label}_desc_{desc_label}_rand_{rand_label}_parent_{parent_label}_train_{language}_cleaned_new.json'),
                    anc_label=anc_label,
                    sib_label=sib_label,
                    desc_label=desc_label,
                    rand_label=rand_label,
                    parent_label=parent_label,
                    language=language,
                    wordnet_df_subset=wordnet_df_subset,
                    split='train')
    wordnet_df_subset = write_sentences(wordnet_df,
                    tree_ids_dev,
                    os.path.join(save_dir, f'wordnet_bansal_anc_{anc_label}_sib_{sib_label}_desc_{desc_label}_rand_{rand_label}_parent_{parent_label}_dev_{language}_cleaned_new.json'),
                    anc_label=anc_label,
                    sib_label=sib_label,
                    desc_label=desc_label,
                    rand_label=rand_label,
                    parent_label=parent_label,
                    language=language,
                    wordnet_df_subset=wordnet_df_subset,
                    split='dev')
    wordnet_df_subset = write_sentences(wordnet_df,
                    tree_ids_test,
                    os.path.join(save_dir, f'wordnet_bansal_anc_{anc_label}_sib_{sib_label}_desc_{desc_label}_rand_{rand_label}_parent_{parent_label}_test_{language}_cleaned_new.json'),
                    anc_label=anc_label,
                    sib_label=sib_label,
                    desc_label=desc_label,
                    rand_label=rand_label,
                    parent_label=parent_label,
                    language=language,
                    wordnet_df_subset=wordnet_df_subset,
                    split='test')
    wordnet_df_subset.to_csv(os.path.join(data_dir, f'bansal14_trees_{language}_cleaned_new.csv'))
    return


def str2bool(word):
    return word in ['true', 'True', 'TRUE']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--save-dir')
    parser.add_argument('--anc-label', type=int)
    parser.add_argument('--sib-label', type=int)
    parser.add_argument('--desc-label', type=int)
    parser.add_argument('--rand-label', type=int)
    parser.add_argument('--parent-label', type=int)

    args = parser.parse_args()

    for language in ['cat', 'cmn', 'ell', 'eus', 'fin', 'fra', 'hrv', 'ind', 'ita', 'jpn', 'nld', 'pol', 'por', 'slv', 'spa', 'tha', 'zsm']:
        print(f'Generating data for {language}')
        generate_data(args.data_dir, args.save_dir, anc_label=args.anc_label, sib_label=args.sib_label, desc_label=args.desc_label, rand_label=args.rand_label, parent_label=args.parent_label, language=language)
