# General utilities such as formatting, getting torch device, and argument conversion

import datetime
import json
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F

from typing import Dict, List

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    return device


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def check_backwards_compatability(args):
    """
    Some parameters were intialized with their experiment name,
    raise warnings if there is inconsistency.
    """
    cased = "cased" in args.experiment_name and "uncased" not in args.experiment_name

    if args.cased != cased:
        print(
            "Warning: explicitly set parameters that override defaults if rerunning experiments!"
        )


def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def read_taxonomy(taxonomy_name: str, file_directory: str):
    """Return dataframe of taxonomy info in file.
    """
    data = pd.read_csv(
        os.path.join(file_directory, f"{taxonomy_name}.taxo"),
        delimiter="\t",
        names=["ID", "term", "hypernym"],
        index_col="ID",
        dtype=str,
        keep_default_na=False,
    )
    return data.drop_duplicates()


def softmax_temp(x, T=1):
    x = np.array(x) / T
    return np.exp(x) / np.sum(np.exp(x))


def get_mean_val(subtree_info, key):
    vals = [val[key] for val in subtree_info.values()]
    return round(np.mean(vals), 2)


def get_wordnet_df(wordnet_filepath, wordnet_column_names, kwargs={}):
    if 'bansal' in wordnet_filepath:
        wordnet_df = pd.read_csv(wordnet_filepath,
            delimiter=",",
            names=wordnet_column_names,
            dtype=str,
            keep_default_na=False,
            **kwargs
        )
    else:
        print('loading df in semeval format')
        wordnet_df = pd.read_csv(wordnet_filepath,
            delimiter=",",
            header=0,
            dtype=str,
            keep_default_na=False,
            **kwargs
        )
    return wordnet_df


def get_sentences_results_wordnet_df(wordnet_filepath,
        results_filepath,
        sentences_filepath,
        wordnet_column_names = ['hypernym', 'term', 'tree_id', 'train_test_split']):
    wordnet_df = get_wordnet_df(wordnet_filepath, wordnet_column_names)
    with open(results_filepath, "r") as f:
        results = json.load(f)

    with open(sentences_filepath, "r") as f:
        sentences = json.load(f)

    return sentences, results, wordnet_df


def print_average_metrics(subtrees_info: Dict,
        epoch_num: int,
        keys_to_print = ['pruned_precision', 'pruned_recall', 'pruned_f1']):
    
    values_print_string = ' // '.join([f'{key} {get_mean_val(subtrees_info, key)}' for key in keys_to_print])
    print(f'Epoch num {epoch_num}: {values_print_string}')
