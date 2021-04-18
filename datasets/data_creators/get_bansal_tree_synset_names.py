import os
import pandas as pd

from nltk.corpus import wordnet as wn


def clean_term(term):
    return term.replace('_$_', '_')


def get_hyponym_lemma_names(synset):
    hyponym_lemma_names = []
    for hyponym in synset.hyponyms():
        hyponym_lemma_names += hyponym.lemma_names()
    return hyponym_lemma_names


def fill_in_synsets(wordnet_df_subset, parent_synset, terms, synsets_dict):
    if len(terms) == 0:
        return
    hyponym_synsets = parent_synset.hyponyms()
    for term in terms:
        term_children = wordnet_df_subset[wordnet_df_subset.hypernym == term]['term']

        # Get the synset for this term.
        possible_synsets = []
        for synset in hyponym_synsets:
            if clean_term(term) in synset.lemma_names():
                possible_synsets.append(synset)
        if len(possible_synsets) > 1:
            if len(term_children) > 0:
                term_children_cleaned = [clean_term(child) for child in term_children]
                possible_synsets_subset = []
                for synset in possible_synsets:
                    hyponym_lemma_names = get_hyponym_lemma_names(synset)
                    if len(set(hyponym_lemma_names).intersection(term_children_cleaned)) == len(term_children_cleaned):
                        possible_synsets_subset.append(synset)
                possible_synsets = possible_synsets_subset
            else:
                print(f'Cannot resolve ambiguity for term {term} with parent synset {parent_synset}')
        elif len(possible_synsets) == 0:
            possible_synsets = wn.synsets(clean_term(term))
            print(f'No matching synsets for term {term} with parent synset {parent_synset}')
            if len(possible_synsets) > 1:
                print('TODO: Manually correct this synset :(')
        assigned_synset = possible_synsets[0]
        synsets_dict[term] = assigned_synset.name()

        # Recursively get synsets for children.
        fill_in_synsets(wordnet_df_subset, parent_synset=assigned_synset, terms=term_children, synsets_dict=synsets_dict)


def get_synset_names_from_tree_df(wordnet_df_subset):
    synsets_dict = {}
    hypernyms = list(set(wordnet_df_subset.hypernym))
    hypernym_has_parent = [hypernym in list(wordnet_df_subset.term) for hypernym in hypernyms]
    root = hypernyms[hypernym_has_parent.index(False)]

    # Get the children of the root.
    root_df_subset = wordnet_df_subset[wordnet_df_subset.hypernym == root]
    hyponyms = [clean_term(term) for term in root_df_subset['term']]

    # Get the synset with the correct hyponyms.
    possible_synsets = wn.synsets(clean_term(root))
    max_hyponyms_matched = 0
    matching_synsets = []
    for synset in possible_synsets:
        hyponym_lemma_names = get_hyponym_lemma_names(synset)
        num_hyponyms_matched = len(set(hyponym_lemma_names).intersection(hyponyms))
        if num_hyponyms_matched == max_hyponyms_matched:
            matching_synsets.append(synset)
        elif num_hyponyms_matched > max_hyponyms_matched:
            max_hyponyms_matched = num_hyponyms_matched
            matching_synsets = [synset]
    # Check that exactly one synset fulfills this constraint.
    if len(matching_synsets) != 1:
        root_grandchildren = [clean_term(grandchild) for grandchild in wordnet_df_subset[wordnet_df_subset.hypernym.isin(list(root_df_subset['term']))]['term']]
        matching_synsets_subset = []
        for synset in matching_synsets:
            synset_grandchildren_lemma_names = []
            for child_synset in synset.hyponyms():
                synset_grandchildren_lemma_names += get_hyponym_lemma_names(child_synset)
            if len(set(synset_grandchildren_lemma_names).intersection(root_grandchildren)) == len(root_grandchildren):
                matching_synsets_subset.append(synset)
        if len(matching_synsets_subset) != 1:
            print(f'NOTE: AMBIGUOUS SYNSET: {matching_synsets_subset}')
        matching_synsets = matching_synsets_subset
    assigned_synset = matching_synsets[0]
    synsets_dict[root] = assigned_synset.name()
    fill_in_synsets(wordnet_df_subset, parent_synset=assigned_synset, terms=root_df_subset['term'], synsets_dict=synsets_dict)
    return synsets_dict


def save_synset_names_df(data_dir='df_csvs'):
    english_df_column_names = ['hypernym', 'term', 'tree_id', 'train_test_split']
    wordnet_df = pd.read_csv(os.path.join(data_dir, 'bansal14_trees.csv'), delimiter=',',
                             names=english_df_column_names,
                             dtype=str, keep_default_na=False)

    data_dict = {'hypernym': [],
            'term': [],
            'tree_id': [],
            'train_test_split': []}

    tree_ids = set(wordnet_df.tree_id)
    for idx, tree_id in enumerate(tree_ids):
        print(f'Processed {idx} of {len(tree_ids)} rows')
        wordnet_df_subset = wordnet_df[wordnet_df.tree_id == tree_id]
        synset_names_dict = get_synset_names_from_tree_df(wordnet_df_subset)
        for idx, row in wordnet_df_subset.iterrows():
            term, hypernym = row['term'], row['hypernym']
            data_dict['hypernym'].append(synset_names_dict[hypernym])
            data_dict['term'].append(synset_names_dict[term])
            data_dict['tree_id'].append(row['tree_id'])
            data_dict['train_test_split'].append(row['train_test_split'])

    wordnet_df_multilingual = pd.DataFrame(data_dict)

    wordnet_df_multilingual.to_csv(os.path.join(data_dir, 'bansal14_trees_synset_names_cleaned.csv'))


if __name__ == '__main__':
    save_synset_names_df()
