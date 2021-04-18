import json
import tqdm
import pandas as pd
import re

import argparse
import logging

import nltk
from nltk.corpus import wordnet as wn

from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups

from typing import Union, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from wiktionaryparser import WiktionaryParser
from bs4 import BeautifulSoup
import wikipedia
import requests

def get_context_wikipedia(term, max_length=400):
    try:
        context = wikipedia.summary(term)[:max_length]
        return context
    except:
        return None

def get_context_wiktionary(term: str, tree_terms: List[str]) -> Union[None, str]:
    # vectorizer = TfidfVectorizer()
    definitions = None
    try:
        word = parser.fetch(term)
        if word and word[0]['definitions']:
            definitions = word[0]['definitions'][0]['text'] 
            # definitions_tfidf  = vectorizer.fit_transform(definitions)
            return definitions
    
    except:
        return definitions

def get_context_merriam_webster(term):
    definitions = None
    try:
        page = requests.get(f"https://www.merriam-webster.com/dictionary/{term}")
        soup = BeautifulSoup(page.content, 'html.parser')
        definition_divs = soup.findAll("span", {"class": "dtText"})
        definitions = [definition_div.getText() for definition_div in definition_divs]
        return definitions
    except:
        return definitions

def create_context_file(cached_contexts_filename, new_contexts_filename, wordnet_trees_file):
    with open(cached_contexts_filename) as json_file:
        contexts = json.load(json_file)

    parser = WiktionaryParser()

    wordnet_df = pd.read_csv(
        wordnet_trees_file,
        delimiter=",",
        dtype={'term1': str, 'term2': str, 'tree_id': int},
        keep_default_na=False,
        header=None,
        names=['term1', 'term2', 'tree_id', 'data_split'],
    )

    num_terms_total = 0
    num_terms_without_wiktionary_context = 0
    num_terms_without_wikipedia_context = 0
    num_terms_without_merriam_webster_context = 0
    num_terms_without_any_context = 0

    for tree_id, group in tqdm.tqdm(wordnet_df.groupby('tree_id')):
        terms = list(set(group['term1'].tolist() + group['term2'].tolist()))
        for term in terms: 
            num_terms_total += 1
            term = re.sub(r'(_\$_)|_', ' ', term)


            wikipedia_context = None
            wiktionary_context = None
            merriam_webster_context = None
                        
            if contexts.get(term):
                wikipedia_context = contexts.get(term).get('wikipedia')
                if not wikipedia_context:
                    wikipedia_context = get_context_wikipedia(term)
                
                wiktionary_context = contexts.get(term).get('wiktionary')
                if not wiktionary_context:
                    wiktionary_context = get_context_wiktionary(term, terms)

                
            if not wikipedia_context and not wiktionary_context:
                merriam_webster_context = get_context_merriam_webster(term)
                        
            # Keep track of number of missing contexts
            if not wiktionary_context:
                num_terms_without_wiktionary_context += 1
            
            if not wikipedia_context:
                num_terms_without_wikipedia_context += 1

            if not merriam_webster_context:
                num_terms_without_merriam_webster_context += 1

            if not wiktionary_context and not wikipedia_context and not merriam_webster_context:
                num_terms_without_any_context += 1
            
            contexts[term] = {'wiktionary': wiktionary_context, 'wikipedia': wikipedia_context, 'merriam_webster': merriam_webster_context}
        if tree_id % 10 == 0:
            print(contexts[term])


    print(f"Total terms: {num_terms_total}")
    print(f"Wiktionary misses: {num_terms_without_wiktionary_context}")
    print(f"Wikipedia misses: {num_terms_without_wikipedia_context}")
    print(f"Merriam webster misses: {num_terms_without_merriam_webster_context}")
    print(f"All miss : {num_terms_without_any_context}")
        
    with open(new_contexts_filename, "w") as outfile:
        json.dump(contexts, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-contexts-filename", type=str)
    parser.add_argument("--new-contexts-filename", type=str)
    parser.add_argument("--wordnet-trees-file", type=str)

    args = parser.parse_args()
    create_context_file(args.cached_contexts_filename, args.new_contexts_filename, args.wordnet_trees_file)
