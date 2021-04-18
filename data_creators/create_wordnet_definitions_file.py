import json

from nltk.corpus import wordnet as wn
from ctp.utils import get_wordnet_df

definitions_filename = './definitions_bansal_new.json'
wordnet_synsets_filename = '../datasets/data_creators/df_csvs/bansal14_trees_synset_names_cleaned_new.csv'
wordnet_column_names = ['hypernym', 'term', 'tree_id', 'train_test_split']
wordnet_df = get_wordnet_df(wordnet_synsets_filename, wordnet_column_names, {'header': 0})

definitions_dict = {}
tree_ids = set(wordnet_df.tree_id)

for tree_id in tree_ids:
    tree_df = wordnet_df[wordnet_df.tree_id == tree_id]
    synsets = set(tree_df.term).union(tree_df.hypernym)
    definitions_dict[tree_id] = {wn.synset(synset).lemma_names()[0].replace('_', ' '): wn.synset(synset).definition()
            for synset in synsets}

with open(definitions_filename, 'w') as f:
    json.dump(definitions_dict, f)
