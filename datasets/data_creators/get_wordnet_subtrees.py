"""
Replicate data from "Structured Learning for Taxonomy Induction with Belief Propagation"
Bansal et al.
"""
import argparse

from typing import Any, Tuple, Set, List
from tqdm import tqdm

from nltk.corpus import wordnet as wn


def get_ancestor_at_height(synset: Any, height: int) -> Any:
    """
    Return synset of ``height`` or None if the tree is of height
    ``height``
    """
    for _ in range(height):
        hypernyms = synset.hypernyms()
        if hypernyms:
            synset = hypernyms[0]
            _, depth = get_tree_descendents(synset, set())
            if depth == height:
                return synset
    return None


def get_tree_descendents(synset: Any, descendents: Set[Any]) -> Set[Any]:
    hyponyms = synset.hyponyms()
    descendents.add(synset)
    if not hyponyms:
        return descendents, 0
    else:
        children_depths = []
        depth = 0
        for hyponym in hyponyms:
            if hyponym not in descendents:
                tree_descendents, tree_depth = get_tree_descendents(
                    hyponym, descendents
                )
                descendents |= tree_descendents
                children_depths.append(tree_depth)
                descendents.add(hyponym)
        if children_depths:
            depth = max(children_depths) + 1
        return descendents, depth


def tree_to_info(tree: Any) -> List[Tuple[Any, Any]]:
    hyponyms = tree.hyponyms()
    if not hyponyms:
        return [], set()
    else:
        relations = []
        synsets = set()
        for hyponym in hyponyms:
            relations.append((tree, hyponym))
            synsets.add(tree)
            synsets.add(hyponym)
            child_relations, child_synsets = tree_to_info(hyponym)
            relations.extend(child_relations)
            synsets = synsets.union(child_synsets)
        return relations, synsets


def write_trees_to_file(relations_filename: str,
                        synsets_filename: str,
                        trees: List[Any],
                        synset_delimiter: str = ';')-> None:
    print(f"Writing trees to {relations_filename}")
    relations_file = open(relations_filename, "w")
    synsets_file = open(synsets_filename, "w")
    for tree_id, tree in enumerate(trees):
        relations, synsets = tree_to_info(tree)
        for relation in relations:
            relations_file.write(
                f"{relation[0].lemma_names()[0]},{relation[1].lemma_names()[0]},{tree_id}\n"
            )
        for synset in synsets:
            synsets_file.write(
                f"{synset.lemma_names()[0]}{synset_delimiter}{synset.name()}{synset_delimiter}{synset.definition().replace(';', ',')}{synset_delimiter}{tree_id}\n"
            )
    relations_file.close()
    synsets_file.close()


if __name__ == "__main__":
    """
    extracted from WordNet 3.0 all bottomed-out full subtrees which had a
    tree-height of 3 (i.e., 4 nodes from root to leaf), and contained (10, 50] terms.
    """
    seen_synset_values = set()
    trees = []

    min_size = 10
    max_size = 50
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-type",
        help='if "term", removes repeated terms. if "sense", removes repeated senses.',
        default="term",
        choices=["term", "sense"],
    )
    args = parser.parse_args()
    relations_filename = f"wordnet_trees_remove_repeated_{args.split_type}.csv"
    synsets_filename = f"wordnet_trees_remove_repeated_{args.split_type}_synsets.csv"

    for synset_id, synset in enumerate(tqdm(list(wn.all_synsets()))):
        ancestor = get_ancestor_at_height(synset, 3)
        if ancestor:
            if args.split_type == "term":
                ancestor_name = ancestor.lemma_names()[0]
            elif args.split_type == "sense":
                ancestor_name = ancestor.name()
            if ancestor_name in seen_synset_values:
                continue
            descendents, depth = get_tree_descendents(ancestor, set())

            if args.split_type == "term":
                descendent_values = set(
                    [descendent.lemma_names()[0] for descendent in descendents]
                )
            elif args.split_type == "sense":
                descendent_values = set(
                    [descendent.name() for descendent in descendents]
                )
            """
            if descendent_values & seen_synset_values:
                continue
            """

            if len(
                set([descendent.lemma_names()[0] for descendent in descendents])
            ) != len([descendent.lemma_names()[0] for descendent in descendents]):
                continue

            if len(descendents) <= min_size or len(descendents) > max_size:
                continue

            print(
                f"Adding tree {ancestor} of size {len(descendents)} and depth {depth}"
            )
            trees.append(ancestor)
            seen_synset_values.add(ancestor_name)
            seen_synset_values = seen_synset_values | descendent_values

    print(f"Found {len(trees)} trees of size between {min_size} and {max_size}")
    write_trees_to_file(relations_filename=relations_filename,
                        synsets_filename=synsets_filename,
                        trees=trees)
