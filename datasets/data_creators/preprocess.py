'''
Data Preprocessing
Adapted from: https://github.com/IBM/gnn-taxo-construction/blob/7ebaaf5ec856caf47a7b26a5d5da462f9b37fd17/preprocess.py
'''

from __future__ import division
from __future__ import print_function
from ctp.datasets.data_creators.shang2020_utils_tree import read_tree_file
import codecs
import os
import pandas as pd
import re
import os.path


# Read the Semeval Taxonomies
def read_files(in_path, given_root=False, filter_root=False, allow_up=True, noUnderscore=False):
    trees = []
    for root, dirs, files in os.walk(in_path):
        for filename in files:
            if not filename.endswith('taxo'):
                continue
            file_path = root + filename
            print('read_edge_files', file_path)
            with codecs.open(file_path, 'r', 'utf-8') as f:
                hypo2hyper_edgeonly = []
                terms = set()
                for line in f:
                    hypo, hyper = line.strip().lower().split('\t')[1:]
                    hypo_ = re.sub(' ', '_', hypo)
                    hyper_ = re.sub(' ', '_', hyper)
                    terms.add(hypo_)
                    terms.add(hyper_)
                    hypo2hyper_edgeonly.append([hypo_, hyper_])
            print(len(terms))

            trees.append([terms, hypo2hyper_edgeonly, filename])
    return trees


# Load the data from dataset_file
def load_dataset(dataset_file, relations):
    """
    Loads a dataset file
    :param dataset_file: the file path
    :return: a list of dataset instances, (x, y, relation)
    """
    with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
        dataset = [tuple(line.strip().split('\t')) for line in f_in]
        dataset = {(x.lower(), y.lower()): relation for (x, y, relation) in dataset if relation in relations}
    return dataset


################### Get the labels/taxonomies ######################
# Read all taxonomies
trees, trees_train_relations = read_tree_file(
    "../data_creators/bansal-taxo-generalsetup/wn-bo-trees-4-11-50-train533.ptb",
    tree_start_index=0, data_split='train', given_root=False, filter_root=False, allow_up=True)
trees_val, trees_val_relations = read_tree_file(
    "../data_creators/bansal-taxo-generalsetup/wn-bo-trees-4-11-50-dev114.ptb",
    tree_start_index=533, data_split='dev', given_root=False, filter_root=False, allow_up=True)
trees_test, trees_test_relations = read_tree_file(
    "../data_creators/bansal-taxo-generalsetup/wn-bo-trees-4-11-50-test114.ptb",
    tree_start_index=533 + 114, data_split='test', given_root=False, filter_root=False, allow_up=True)
trees_semeval = read_files('../data_creators/semeval_2016_task_13/TExEval-2_testdata_1.2/gs_taxo/EN/',
                           given_root=True, filter_root=False, allow_up=False)
trees_semeval_trial = read_files("../data_creators/semeval_2016_task_13/TExEval_trialdata_1.2/EN/",
                                 given_root=True, filter_root=False, allow_up=False)

# Build the vocabulary
vocab = set()
for i in range(len(trees)):
    vocab = vocab.union(trees[i].terms)
for i in range(len(trees_val)):
    vocab = vocab.union(trees_val[i].terms)
for i in range(len(trees_test)):
    vocab = vocab.union(trees_test[i].terms)
print('size of terms in training:', len(vocab))

for i in range(len(trees_semeval)):
    vocab = vocab.union(trees_semeval[i][0])
print('size of terms in the semeval:', len(vocab))
for i in range(len(trees_semeval_trial)):
    vocab = vocab.union(trees_semeval_trial[i][0])
print('size of terms added trial:', len(vocab))

vocab_semeval = set()
for i in range(len(trees_semeval)):
    vocab_semeval = vocab_semeval.union(trees_semeval[i][0])
print('size of terms (semeval):', len(vocab_semeval))

# Remove the overlapping taxonomies.
tree_no_intersect = []
count = 0
falsecount = 0
excluded_trees = []


def convert_to_semeval_format(terms_set):
    return set([term.lower().replace('_$_', '_') for term in terms_set])


for i in range(len(trees)):
    if len(convert_to_semeval_format(trees[i].terms) & vocab_semeval) == 0:
        count = count + 1
        tree_no_intersect.append(trees[i])
    else:
        falsecount = falsecount + 1
        excluded_terms = set(trees[i].terms)
        excluded_trees.append(excluded_terms)

for i in range(len(trees_val)):
    if len(convert_to_semeval_format(trees_val[i].terms) & vocab_semeval) == 0:
        count = count + 1
        tree_no_intersect.append(trees_val[i])
    else:
        falsecount = falsecount + 1
        excluded_terms = set(trees_val[i].terms)
        excluded_trees.append(excluded_terms)

for i in range(len(trees_test)):
    if len(convert_to_semeval_format(trees_test[i].terms) & vocab_semeval) == 0:
        count = count + 1
        tree_no_intersect.append(trees_test[i])
    else:
        falsecount = falsecount + 1
        excluded_terms = set(trees_test[i].terms)
        excluded_trees.append(excluded_terms)
print('num of trees which has no intersaction with label taxos:', count)
print('Trees need to be removed:', falsecount)

### Save bansal 2014 trees.
data_splits_and_tree_start_indices = {'train': {'num_trees': 533, 'tree_start_index': 0},
        'dev': {'num_trees': 114, 'tree_start_index': 533},
        'test': {'num_trees': 114, 'tree_start_index': 647}}

df = pd.DataFrame(trees_train_relations + trees_val_relations + trees_test_relations)
df.to_csv('./df_csvs/bansal14_trees.csv', index=False, header=False)


### Save bansal 2014 trees without semeval terms.
df_no_semeval = pd.DataFrame()
df.columns = ['hypernym', 'term', 'tree_id', 'data_split']
num_included = 0
for tree_id in set(df.tree_id):
    tree_df = df[df.tree_id == tree_id]
    tree_terms = set(tree_df.term).union(tree_df.hypernym)
    if tree_terms not in excluded_trees:
        num_included += 1
        df_no_semeval = df_no_semeval.append(tree_df, ignore_index=True)
print(f'Num included trees: {num_included}')
df_no_semeval.to_csv('./df_csvs/bansal14_trees_no_semeval.csv', index=False, header=False)
