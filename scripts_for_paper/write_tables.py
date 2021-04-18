import json
import numpy as np

from ctp.utils import get_wordnet_df

language_to_learning_rate_dict = {
        'cat': {'calbert': 5, 'mbert': 6},
        'cmn': {'chinese': 6, 'mbert': 6},
        'fin': {'finbert': 6, 'mbert': 6},
        'fra': {'fr_bert': 6, 'mbert': 6},
        'ita': {'it_bert': 5, 'mbert': 6},
        'nld': {'nl_bert': 5, 'mbert': 5},
        'pol': {'polbert': 5, 'mbert': 5},
        'por': {'portuguese': 6, 'mbert': 5},
        'spa': {'spanish': 6, 'mbert': 5},
}

model_to_learning_rate_dict = {
        'bert': 6,
        'roberta_large': 6,
        'roberta': 6,
        'bert_large': 6,
        'retrieval_roberta_large': 6,
        }

language_to_model_name_dict = {
    'cat': 'calbert',
    'cmn': 'chinese',
    'fin': 'finbert',
    'fra': 'fr_bert',
    'ita': 'it_bert',
    'nld': 'nl_bert',
    'pol': 'polbert',
    'por': 'portuguese',
    'spa': 'spanish'
}
language_names_dict = {
    'cat': 'Catalan',
    'cmn': 'Chinese',
    'eng': 'English',
    'fin': 'Finnish',
    'fra': 'French',
    'ita': 'Italian',
    'nld': 'Dutch',
    'pol': 'Polish',
    'por': 'Portuguese',
    'spa': 'Spanish'
}
language_id_to_written_model_name_dict = {
    'cat': 'Calbert',
    'cmn': 'Chinese BERT',
    'fin': 'FinBERT',
    'fra': 'French Europeana BERT',
    'ita': 'Italian BERT',
    'nld': 'BERTje',
    'pol': 'Polbert',
    'por': 'BERTimbau',
    'spa': 'BETO'
}

model_written_names_dict = {
        'bert': 'BERT',
        'roberta_large': 'RoBERTa Large',
        'bert_large': 'BERT Large',
        'retrieval_roberta_large': 'RoBERTa Large (Retrieval)',
        'wordnet_defs_roberta_large': 'RoBERTa Large (Wordnet Definitions)'
}


def get_merged_results_dicts(results_filepaths):
    results_dict = {}
    for results_filepath in results_filepaths:
        with open(results_filepath, 'r') as f:
            results = json.load(f)
        for model_name, model_results in results.items():
            if model_name not in results_dict or model_results['pruned_f1'] > results_dict[model_name]['pruned_f1']:
                results_dict[model_name] = model_results
    return results_dict


def write_mling_learning_rates_table(results_filepaths=['../outputs/dev_results_dict_nlp1.json', '../outputs/dev_results_dict_nlp2_new.json'],
        results_key='pruned_f1',
        lrs=[5, 6, 7]):
    language_ids = list(language_to_model_name_dict.keys())
    language_ids.sort()
    results = {}
    for results_filepath in results_filepaths:
        with open(results_filepath, 'r') as f:
            results = {**results, **json.load(f)}
    fname = './table_files/mling_learning_rates.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{llrrr}\n')
    f.write('\\toprule\n')
    f.write('\tLanguage & Model & 1e-5 & 1e-6 & 1e-7 \\\\')
    for language_id in language_ids:
        language = language_names_dict[language_id]
        language_model_name = language_id_to_written_model_name_dict[language_id]
        language_ckpt_name = 'ckpt_par_{language}_{language_model_name}_'.format(language=language_id, language_model_name=language_to_model_name_dict[language_id]) + '1e{lr}.json'
        language_mbert_ckpt_name = 'ckpt_par_{language}_{language_model_name}_'.format(language=language_id, language_model_name='mbert') + '1e{lr}.json'
        lr_results = {lr: '%0.1f' % (results[language_ckpt_name.format(lr=lr)][results_key] * 100) if language_ckpt_name.format(lr=lr) in results else 0 for lr in lrs}
        max_f1 = str(max(lr_results.values()))
        f.write('\\midrule\n\t\\multirow{2}{*}{' + language + '}' + f' & {language_model_name} & {lr_results[5]} & {lr_results[6]} & {lr_results[7]} \\\\'.replace(max_f1, '\\textbf{' + max_f1 + '}'))
        lr_results = {lr: results[language_mbert_ckpt_name.format(lr=lr)][results_key] * 100 if language_ckpt_name.format(lr=lr) in results else 0 for lr in lrs}
        max_f1 = str(max(lr_results.values()))
        f.write(f'\n\t & mBERT & {"%0.1f" % lr_results[5]} & {"%0.1f" % lr_results[6]} & {"%0.1f" % lr_results[7]} \\\\'.replace(max_f1, '\\textbf{' + max_f1 + '}'))
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_mling_results_table(results_filepaths=['../outputs/test_results_dict_nlp2_new.json', '../outputs/test_results_dict_nlp1.json'],
        results_keys=['pruned_precision', 'pruned_recall', 'pruned_f1'],
        seed_strings=['', '_seed1', '_seed2']):
    results_dict = get_merged_results_dicts(results_filepaths)
    language_ids = list(language_to_model_name_dict.keys())
    language_ids.sort()

    fname = './table_files/mling_results.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{llrrr}\n')
    f.write('\\toprule\n')
    f.write('\tLanguage & Model & \\textbf{P} & \\textbf{R} & \\textbf{F1} \\\\')
    for language_id in language_ids:
        language = language_names_dict[language_id]
        language_model_name = language_id_to_written_model_name_dict[language_id]
        language_model_name_ckpt = language_to_model_name_dict[language_id]
        # Get results.
        language_ckpt_names = [f'ckpt_par_{language_id}_{language_model_name_ckpt}_1e{language_to_learning_rate_dict[language_id][language_model_name_ckpt]}{seed_string}.json' for seed_string in seed_strings]
        language_mbert_ckpt_names = [f'ckpt_par_{language_id}_mbert_1e{language_to_learning_rate_dict[language_id]["mbert"]}{seed_string}.json' for seed_string in seed_strings]
        language_results = {results_key: '%0.1f' % (100 * np.mean([results_dict[language_ckpt_name][results_key] for language_ckpt_name in language_ckpt_names])) for results_key in results_keys}
        mbert_results = {results_key: '%0.1f' % (100 * np.mean([results_dict[language_ckpt_name][results_key] if language_ckpt_name in results_dict else 'NONE' for language_ckpt_name in language_mbert_ckpt_names])) for results_key in results_keys}
        # Write results.
        f.write('\\midrule\n\t\\multirow{2}{*}{' + language + '}' + f' & {language_model_name} & {language_results["pruned_precision"]} & {language_results["pruned_recall"]} & {language_results["pruned_f1"]} \\\\')
        f.write(f'\n\t & mBERT & {mbert_results["pruned_precision"]} & {mbert_results["pruned_recall"]} & {mbert_results["pruned_f1"]} \\\\')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_en_learning_rates_table(results_filepaths=['../outputs/dev_results_dict_nlp2_new.json', '../outputs/dev_results_dict_nlp1.json'],
        results_key='pruned_f1',
        lrs=[5, 6, 7]):
    model_names = list(model_to_learning_rate_dict.keys())
    model_names.sort()
    results_dict = get_merged_results_dicts(results_filepaths)
    fname = './table_files/en_learning_rates.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{lrrr}\n')
    f.write('\\toprule\n')
    f.write('\t Model & 1e-5 & 1e-6 & 1e-7 \\\\')
    for model_name in model_names:
        model_ckpt_name = f'ckpt_par_{model_name}_' + '1e{lr}_seed0.json'
        lr_results = {lr: '%0.1f' % (results_dict[model_ckpt_name.format(lr=lr)][results_key] * 100) if model_ckpt_name.format(lr=lr) in results_dict else 'TODO' for lr in lrs}
        max_f1 = str(max(lr_results.values()))
        f.write(f'\\midrule\n\t{model_written_names_dict[model_name]} & {lr_results[5]} & {lr_results[6]} & {lr_results[7]} \\\\'.replace(max_f1, '\\textbf{' + max_f1 + '}'))
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_dataset_statistics_table():
    fname = './table_files/dataset_statistics.txt'
    columns = ['hypernym', 'term', 'tree_id', 'train_test_split']
    f = open(fname, 'w')
    f.write('\\begin{table*}[thb!]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{lrrrrrr}\n')
    f.write('\\toprule\n')
    f.write('& \\multicolumn{3}{c}{\\textbf{Num Trees}} & \\multicolumn{3}{c}{\\textbf{Mean Nodes per Tree}}\\\\')
    f.write('& Train & Dev & Test & Train & Dev & Test')
    language_ids = list(language_names_dict.keys())
    language_ids.sort()
    for language_id in language_ids:
        if language_id == 'eng':
            wordnet_df = get_wordnet_df('../datasets/data_creators/df_csvs/bansal14_trees.csv', columns, {'header': 0})
        else:
            wordnet_df = get_wordnet_df(f'../datasets/data_creators/df_csvs/bansal14_trees_{language_id}_cleaned.csv', columns, {'header': 0})
        num_trees_dict = {}
        tree_size_dict = {}
        for split in ['train', 'dev', 'test']:
            df_subset = wordnet_df[wordnet_df['train_test_split'] == split]
            tree_ids = set(df_subset.tree_id)
            num_trees_dict[split] = len(tree_ids)
            total_num_nodes = 0
            for tree_id in tree_ids:
                tree_id_subset = df_subset[df_subset.tree_id == tree_id]
                total_num_nodes += len(set(tree_id_subset.term).union(tree_id_subset.hypernym))
            tree_size_dict[split] = total_num_nodes / len(tree_ids)
        f.write(f'\\\\  \\midrule\n {language_names_dict[language_id]} & {num_trees_dict["train"]} & {num_trees_dict["dev"]} & {"%0.1f" % num_trees_dict["test"]} & {"%0.1f" % tree_size_dict["train"]} & {"%0.1f" % tree_size_dict["dev"]} & {"%0.1f" % tree_size_dict["test"]}')
    f.write('\\\\ \\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_mling_results_table_trial_separated(results_filepaths=['../outputs/test_results_dict_nlp2_new.json', '../outputs/test_results_dict_nlp1.json'],
        results_key='pruned_f1',
        seed_strings=['', '_seed1', '_seed2']):
    results_dict = get_merged_results_dicts(results_filepaths)
    language_ids = list(language_to_model_name_dict.keys())
    language_ids.sort()

    fname = './table_files/mling_results_trial_separated.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{llrrr}\n')
    f.write('\\toprule\n')
    f.write('\tLanguage & Model & \\textbf{Trial 0} & \\textbf{Trial 1} & \\textbf{Trial 2} \\\\')
    for language_id in language_ids:
        language = language_names_dict[language_id]
        language_model_name = language_id_to_written_model_name_dict[language_id]
        language_model_name_ckpt = language_to_model_name_dict[language_id]
        # Get results.
        language_ckpt_names = [f'ckpt_par_{language_id}_{language_model_name_ckpt}_1e{language_to_learning_rate_dict[language_id][language_model_name_ckpt]}{seed_string}.json' for seed_string in seed_strings]
        language_mbert_ckpt_names = [f'ckpt_par_{language_id}_mbert_1e{language_to_learning_rate_dict[language_id]["mbert"]}{seed_string}.json' for seed_string in seed_strings]
        language_results = {i: '%0.1f' % (100 * results_dict[language_ckpt_name][results_key]) if language_ckpt_name in results_dict else 'NONE' for i, language_ckpt_name in enumerate(language_ckpt_names)}
        mbert_results = {i: '%0.1f' % (100 * results_dict[language_ckpt_name][results_key]) if language_ckpt_name in results_dict else 'NONE' for i, language_ckpt_name in enumerate(language_mbert_ckpt_names)}
        # Write results.
        f.write('\\midrule\n\t\\multirow{2}{*}{' + language + '}' + f' & {language_model_name} & {language_results[0]} & {language_results[1]} & {language_results[2]} \\\\')
        f.write(f'\n\t & mBERT & {mbert_results[0]} & {mbert_results[1]} & {mbert_results[2]} \\\\')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_en_results_table_trial_separated(results_filepaths=['../outputs/test_results_dict_nlp2_new.json', '../outputs/test_results_dict_nlp1.json'],
        results_key='pruned_f1',
        seed_nums=[0, 1, 2]):
    results_dict = get_merged_results_dicts(results_filepaths)
    model_names = list(model_to_learning_rate_dict.keys())
    model_names.sort()

    fname = './table_files/en_results_trial_separated.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{lrrr}\n')
    f.write('\\toprule\n')
    f.write('\tModel & \\textbf{Trial 0} & \\textbf{Trial 1} & \\textbf{Trial 2} \\\\')
    for model_name in model_names:
        lr = model_to_learning_rate_dict[model_name]
        # Get results.
        model_ckpt_name = f'ckpt_par_{model_name}_' + '1e{lr}_seed{seed_num}.json'
        trial_results = {seed_num: '%0.1f' % (results_dict[model_ckpt_name.format(lr=lr, seed_num=seed_num)][results_key] * 100) if model_ckpt_name.format(lr=lr, seed_num=seed_num) in results_dict else 'TODO' for seed_num in seed_nums}
        # Write results.
        f.write('\\midrule\n\t' + model_written_names_dict[model_name] + f' & {trial_results[0]} & {trial_results[1]} & {trial_results[2]} \\\\')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_en_results_table_trial_averages(results_filepaths=['../outputs/test_results_dict_nlp2_new.json', '../outputs/test_results_dict_nlp1.json'],
        results_keys=['pruned_precision', 'pruned_recall', 'pruned_f1'],
        seed_nums=[0, 1, 2]):
    results_dict = get_merged_results_dicts(results_filepaths)
    model_names = list(model_to_learning_rate_dict.keys())
    model_names.sort()

    fname = './table_files/en_results_trial_averages.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{lrrr}\n')
    f.write('\\toprule\n')
    f.write('\tModel & \\textbf{P} & \\textbf{R} & \\textbf{F1} \\\\')
    for model_name in model_names:
        lr = model_to_learning_rate_dict[model_name]
        # Get results.
        model_ckpt_name = f'ckpt_par_{model_name}_' + '1e{lr}_seed{seed_num}.json'
        results = {results_key: '%0.1f' % (100 * np.mean([results_dict[model_ckpt_name.format(lr=lr, seed_num=seed_num)][results_key] for seed_num in seed_nums])) for results_key in results_keys}
        # Write results.
        f.write('\\midrule\n\t' + model_written_names_dict[model_name] + f' & {results["pruned_precision"]} & {results["pruned_recall"]} & {results["pruned_f1"]} \\\\')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_baseline_results_table_trial_separated(results_filepath='./outputs/random_baseline_metrics.json',
        results_key='p',
        seed_nums=[0, 1, 2]):
    with open(results_filepath, 'r') as f:
        results_dict = json.load(f)
    language_ids = list(language_names_dict.keys())
    language_ids.sort()

    fname = './table_files/baseline_results_trial_separated.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{lrrr}\n')
    f.write('\\toprule\n')
    f.write('\tModel & \\textbf{Trial 0} & \\textbf{Trial 1} & \\textbf{Trial 2} \\\\')
    for language_id in language_ids:
        # Get results.
        trial_results = {seed_num: '%0.1f' % (np.mean(results_dict[language_id][str(seed_num)][results_key]) * 100) for seed_num in seed_nums}
        # Write results.
        f.write('\\midrule\n\t' + language_names_dict[language_id] + f' & {trial_results[0]} & {trial_results[1]} & {trial_results[2]} \\\\')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


def write_baseline_results_table_trial_averages(results_filepath='./outputs/random_baseline_metrics.json',
        results_keys=['p', 'r', 'f'],
        seed_nums=[0, 1, 2]):
    with open(results_filepath, 'r') as f:
        results_dict = json.load(f)
    language_ids = list(language_names_dict.keys())
    language_ids.sort()

    fname = './table_files/baseline_results_trial_averages.txt'
    f = open(fname, 'w')
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write('\\begin{tabular}{lrrr}\n')
    f.write('\\toprule\n')
    f.write('\tModel & \\textbf{P} & \\textbf{R} & \\textbf{F1} \\\\')
    for language_id in language_ids:
        # Get results.
        trial_results = {results_key: '%0.1f' % (np.mean([np.mean(results_dict[language_id][str(seed_num)][results_key]) for seed_num in seed_nums]) * 100) for results_key in results_keys}
        # Write results.
        f.write('\\midrule\n\t' + language_names_dict[language_id] + f'& {trial_results["p"]} & {trial_results["r"]} & {trial_results["f"]} \\\\')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}')
    f.close()


if __name__ == '__main__':
    write_mling_learning_rates_table()
    write_dataset_statistics_table()
    write_en_learning_rates_table()
    write_mling_results_table()
    write_mling_results_table_trial_separated()
    write_en_results_table_trial_separated()
    write_en_results_table_trial_averages()
    write_baseline_results_table_trial_averages()
    write_baseline_results_table_trial_separated()
