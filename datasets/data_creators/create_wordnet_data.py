import argparse
import hashlib
import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import sys
import re

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch import nn
import torchtext

sys.path.insert(
    0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir)))
)

from ctp.utils import get_wordnet_df

cos = nn.CosineSimilarity(dim=0, eps=1e-6)
glove = torchtext.vocab.GloVe(name='6B', dim=500)
stop_words = set(stopwords.words('english'))

np.random.seed(2020)

def write_sentences(df, tree_ids, save_filename, anc_label, sib_label, desc_label, rand_label, parent_label, retrieve, contexts, definitions, pos_subset=None):
    """
    df: dataframe containing pair information ('term' and 'hypernym' columns).
    save_filename: filename to save data to.
    """
    print(f'save filename: {save_filename}, {len(tree_ids)} trees')

    def build_graph(df):
        G = nx.DiGraph()
        for idx, row in df.iterrows():
            G.add_edge(row['hypernym'], row['term'])
        return G

    def get_context(term, hypernym, k=1, max_length=400):
        query = f'{term} {hypernym}'
        doc_names, doc_scores = ranker.closest_docs(query, k)
        context = docdb.get_doc_text(doc_names[0])
        return context[:max_length]

    def rank_definitions(term1, term2, definitions_1, definitions_2):
        definitions_1 = [term1 + ' ' + definition for definition in definitions_1]
        definitions_2 = [term2 + ' ' + definition for definition in definitions_2]

        definitions_1_vectors = [sum(glove[word] for word in definition.split(' ') if word not in stop_words) / len(definition.split(' ')) for definition in definitions_1]
        definitions_2_vectors = [sum(glove[word] for word in definition.split(' ') if word not in stop_words) / len(definition.split(' ')) for definition in definitions_2]
        pairs = []
        for idx_1, sentence_vector_1 in enumerate(definitions_1_vectors):
            for idx_2, sentence_vector_2 in enumerate(definitions_2_vectors):
                pairs.append((idx_1, idx_2, abs(cos(sentence_vector_1, sentence_vector_2))))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def rank_definition(term, definitions, bag_of_words_tree):
        definitions_vectors = [sum(glove[word] for word in definition.split(' ') if word not in stop_words) / len(definition.split(' ')) for definition in definitions]

        bag_of_words_vectors = sum(glove[word] for word in bag_of_words_tree if word not in stop_words) / len(bag_of_words_tree)

        definition_idx_tuples = [(idx, definition) for idx, definition in enumerate(definitions_vectors)]
        definition_idx_tuples.sort(key=lambda x: cos(x[1], bag_of_words_vectors), reverse=True)
        contexts = [definition[0] for definition in definition_idx_tuples]
        ranking_scores = [float(cos(definition[1], bag_of_words_vectors).numpy()) for definition in definition_idx_tuples]
        return contexts, ranking_scores

    def get_context_string(term: str, tree_id: int):
        term = re.sub(r'(_\$_)', ' ', term)
        term_context_dict = contexts.get(term, None)

        definitions = []
        if term_context_dict['wiktionary']:
            for idx, definition in enumerate(term_context_dict['wiktionary']):
                if idx != 0:
                    definitions.append(definition)
        if term_context_dict['wikipedia']:
            definitions.append(term_context_dict['wikipedia'])

        if term_context_dict['merriam_webster']:
            definitions.extend(term_context_dict['merriam_webster'])

        terms_in_tree = list(set(df[df['tree_id'] == tree_id]['term']))
        terms_in_tree = [term.replace('_$_', ' ').replace('_', ' ').strip('()') for term in terms_in_tree]

        if definitions:
            rankings, ranking_scores = rank_definition(term, definitions, terms_in_tree)
            term_context_str = ' '.join([definitions[rank] for rank in rankings])
        else:
            term_context_str = ''
            ranking_scores = []

        contexts[term]['context_str'] = term_context_str
        return term_context_str, ranking_scores

    def create_example(pattern, term, hypernym, label, data_dict, tree_id, definitions):
        '''
        args:
            if all_anc_desc=True, make examples with all ancestors and descendants. Else randomly choose one of each.
        '''
        term = term.replace('_$_', ' ')
        hypernym = hypernym.replace('_$_', ' ')
        sentence = pattern.replace('[TERM]', term).replace('[HYPERNYM]', hypernym).replace('_', ' ')
        if retrieve:
            #term_context = contexts[term.replace('_$_', ' ').replace('_', ' ')].get('context_str', '')
            #if not term_context:
            term = term.replace('_$_', ' ').replace('_', ' ')
            term_context, term_context_ranking_scores = get_context_string(term, tree_id)

            # hypernym_context = contexts[hypernym.replace('_$_', ' ').replace('_', ' ')].get('context_str', '')
            hypernym = hypernym.replace('_$_', ' ').replace('_', ' ')
            # if not hypernym_context:
            hypernym_context, hypernym_context_ranking_scores = get_context_string(hypernym, tree_id)
            term_definition = definitions[tree_id][term]
            hypernym_definition = definitions[tree_id][hypernym]
            example = {'sentence': sentence, 'term_context': term_context, 'hypernym_context': hypernym_context, 'label': label, 'tree_id': tree_id, 'term': term, 'hypernym': hypernym, 'term_context_ranking_scores': term_context_ranking_scores, 'hypernym_context_ranking_scores': hypernym_context_ranking_scores, 'term_definition': term_definition, 'hypernym_definition': hypernym_definition}

        else:
            example = {'sentence': sentence, 'label': label, 'tree_id': tree_id, 'term': term, 'hypernym': hypernym}

        example_id = hashlib.sha224(sentence.encode('utf-8')).hexdigest()
        data_dict[example_id] = example
        return

    data_dict = dict()
    for tree_id in tree_ids:
        df_subset = df[df['tree_id'] == tree_id]
        nodes = list(set(df_subset['term']).union(df_subset['hypernym']))

        G = build_graph(df_subset)
        wn_synsets = []
        for node in nodes:
            wn_synsets += wn.synsets(node)
        pos_tags = [synset.pos() for synset in wn_synsets]

        if pos_tags:
            most_common_pos_tag = max(set(pos_tags), key=pos_tags.count)
            if pos_subset and most_common_pos_tag != pos_subset:
                continue

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
                        create_example(pattern, node1, node2, parent_label, data_dict, tree_id, definitions)
                    elif node2 in ancestors:
                        create_example(pattern, node1, node2, anc_label, data_dict, tree_id, definitions)
                    elif node2 in siblings:
                        create_example(pattern, node1, node2, sib_label, data_dict, tree_id, definitions)
                    elif node2 in descendants:
                        create_example(pattern, node1, node2, desc_label, data_dict, tree_id, definitions)
                    else:
                        create_example(pattern, node1, node2, rand_label, data_dict, tree_id, definitions)
        print(f'num examples: {len(data_dict.keys())}')
    with open(save_filename, 'w') as f:
        json.dump(data_dict, f)

    positive_examples = {k: v for k, v in data_dict.items() if v['label'] == 1}
    negative_examples = {k: v for k, v in data_dict.items() if v['label'] == 0}
    sampled_negative_ids = np.random.choice(list(negative_examples.keys()), size=len(positive_examples.keys()), replace=False)
    negative_examples_subset = {k: data_dict[k] for k in sampled_negative_ids}
    balanced_data_dict = {**positive_examples, **negative_examples_subset}
    with open(save_filename.replace('.json', '_balanced.json'), 'w') as f:
        json.dump(balanced_data_dict, f)


def generate_data(data_dir,
                  save_dir,
                  anc_label,
                  sib_label,
                  desc_label,
                  rand_label,
                  parent_label,
                  num_subsampled_trees,
                  retrieve,
                  contexts_filename,
                  definitions_filename,
                  remove_semeval_test_terms=False,
                  pos_subset=None,
                  train_proportion=0.8,
                  dev_vs_test_propotion=0.5,
                  resplit_train_test=False):
    if retrieve:
        # with open('DrQA/contexts_remove_repeated_term.json') as context_file:
        with open(contexts_filename) as context_file:
            contexts = json.load(context_file)
        with open(definitions_filename) as definitions_file:
            definitions = json.load(definitions_file)
    else:
        contexts = None
        definitions = None

    if resplit_train_test:
        wordnet_filepath = os.path.join(data_dir, 'wordnet_trees_remove_repeated_term.csv')
        wordnet_column_names = ['hypernym', 'term', 'tree_id']
        wordnet_df = get_wordnet_df(wordnet_filepath, wordnet_column_names)

        tree_ids = np.unique(wordnet_df['tree_id'])
        if num_subsampled_trees > 0:
            tree_ids = np.random.choice(tree_ids, num_subsampled_trees, replace=False)
        tree_ids_train, tree_ids_test_or_dev = train_test_split(tree_ids, train_size=train_proportion, random_state=2020)
        tree_ids_dev, tree_ids_test = train_test_split(tree_ids_test_or_dev, train_size=dev_vs_test_propotion, random_state=2020)
    elif remove_semeval_test_terms:
        wordnet_filepath = os.path.join(data_dir, 'bansal14_trees_no_semeval.csv')
        wordnet_column_names = ['hypernym', 'term', 'tree_id', 'train_test_split']
        wordnet_df = get_wordnet_df(wordnet_filepath, wordnet_column_names)
        tree_ids = list(set(wordnet_df.tree_id))

        np.random.seed(123)
        tree_ids_train = np.random.choice(tree_ids, int(len(tree_ids) * train_proportion), replace=False)
        tree_ids_dev = [tree_id for tree_id in tree_ids if tree_id not in tree_ids_train]
        wordnet_split = '_bansal_remove_semeval_test'
        print(f'After removing semeval overlap, {len(tree_ids)} trees. ({len(tree_ids_train)} train, {len(tree_ids_dev)} dev).')
    else:
        wordnet_filepath = os.path.join(data_dir, 'bansal14_trees.csv')
        wordnet_column_names = ['hypernym', 'term', 'tree_id', 'train_test_split']
        wordnet_df = get_wordnet_df(wordnet_filepath, wordnet_column_names)
        tree_ids_train = np.unique(wordnet_df[wordnet_df['train_test_split'] == 'train']['tree_id'])
        tree_ids_dev = np.unique(wordnet_df[wordnet_df['train_test_split'] == 'dev']['tree_id'])
        tree_ids_test = np.unique(wordnet_df[wordnet_df['train_test_split'] == 'test']['tree_id'])
        wordnet_split = '_bansal'

    write_sentences(wordnet_df,
                    tree_ids_train,
                    os.path.join(save_dir, f'wordnet{wordnet_split}_anc_{anc_label}_sib_{sib_label}_desc_{desc_label}_rand_{rand_label}_parent_{parent_label}_train_subsample_{num_subsampled_trees}_pos_subset_{pos_subset}.json'),
                    anc_label=anc_label,
                    sib_label=sib_label,
                    desc_label=desc_label,
                    rand_label=rand_label,
                    parent_label=parent_label,
                    retrieve=retrieve,
                    contexts=contexts,
                    definitions=definitions,
                    pos_subset=pos_subset)
    write_sentences(wordnet_df,
                    tree_ids_dev,
                    os.path.join(save_dir, f'wordnet{wordnet_split}_anc_{anc_label}_sib_{sib_label}_desc_{desc_label}_rand_{rand_label}_parent_{parent_label}_dev_subsample_{num_subsampled_trees}_pos_subset_{pos_subset}.json'),
                    anc_label=anc_label,
                    sib_label=sib_label,
                    desc_label=desc_label,
                    rand_label=rand_label,
                    parent_label=parent_label,
                    retrieve=retrieve,
                    pos_subset=pos_subset,
                    definitions=definitions,
                    contexts=contexts)
    if not remove_semeval_test_terms:
        write_sentences(wordnet_df,
                        tree_ids_test,
                        os.path.join(save_dir, f'wordnet{wordnet_split}_anc_{anc_label}_sib_{sib_label}_desc_{desc_label}_rand_{rand_label}_parent_{parent_label}_test_subsample_{num_subsampled_trees}_pos_subset_{pos_subset}.json'),
                        anc_label=anc_label,
                        sib_label=sib_label,
                        desc_label=desc_label,
                        rand_label=rand_label,
                        parent_label=parent_label,
                        retrieve=retrieve,
                        contexts=contexts,
                        definitions=definitions,
                        pos_subset=pos_subset)
    '''
    if retrieve:
            with open(contexts_filename) as context_file:
                contexts = json.load(context_file)
    else:
        contexts = None
    '''

    return


def str2bool(word):
    return word in ['true', 'True', 'TRUE']


def get_semeval_terms(semeval_test_dir: str = './semeval_2016_task_13/TExEval-2_testdata_1.2/gs_terms/EN'):
    semeval_test_terms_filenames = [filename for filename in os.listdir(semeval_test_dir) if '.terms' in filename]
    all_test_terms = set()
    for filename in semeval_test_terms_filenames:
        terms = set(pd.read_csv(os.path.join(semeval_test_dir, filename), delimiter='\t', names=['id', 'term'], keep_default_na=False)['term'])
        all_test_terms = all_test_terms.union(terms)
    print(f'num semeval terms: {len(set(all_test_terms))}')
    all_test_terms = [term.lower() for term in all_test_terms]
    return all_test_terms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir')
    parser.add_argument('--save-dir')
    parser.add_argument('--anc-label', type=int)
    parser.add_argument('--sib-label', type=int)
    parser.add_argument('--desc-label', type=int)
    parser.add_argument('--rand-label', type=int)
    parser.add_argument('--parent-label', type=int)
    parser.add_argument('--remove-semeval-test-terms', action='store_true')
    parser.add_argument('--retrieve', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--pos-subset', type=str, default=None)
    parser.add_argument('--num-subsampled-trees', type=int, default=0, help='num trees to subsample. if 0, uses all trees.')
    parser.add_argument('--resplit-train-test', action='store_true', help='if true, then resplit wordnet scraped wordnet data, otherwise use splits from bansal et al', default=False)
    parser.add_argument('--contexts-filename', type=str, default='../../data_creators/contexts_bansal_1004_with_merriam_webster.json')
    parser.add_argument('--definitions-filename', type=str, default='../../data_creators/definitions_bansal.json')

    args = parser.parse_args()
    generate_data(args.data_dir, args.save_dir, anc_label=args.anc_label, sib_label=args.sib_label, desc_label=args.desc_label, rand_label=args.rand_label, parent_label=args.parent_label, num_subsampled_trees=args.num_subsampled_trees, retrieve=args.retrieve, pos_subset=args.pos_subset, resplit_train_test=args.resplit_train_test, contexts_filename=args.contexts_filename, definitions_filename=args.definitions_filename, remove_semeval_test_terms=args.remove_semeval_test_terms)
