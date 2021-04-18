import argparse
import json
import numpy as np
import os
import pickle
import random
import time
import torch
import tqdm
import subprocess
import sys

from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
from transformers.trainer import set_seed
from typing import List

sys.path.append('.')

from ctp.metrics import compute_metrics, compute_scores, flat_accuracy
from ctp.utils import (
    check_backwards_compatability,
    format_time,
    get_device,
    get_sentences_results_wordnet_df,
    print_average_metrics,
    str2bool,
)
from ctp.inference.examine_subtrees import run_inference_subtree
from datasets.model_inputs.dataset import get_dataset

device = get_device()


def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    set_seed(seed_val)


def run_inference(test_filename: str,
        epoch_num: int,
        results_filename: str,
        subtrees_filename: str,
        results_dir: str,
        data_dir: str,
        wordnet_dir: str = './datasets/data_creators/df_csvs/',
        prediction_metric_type: str = 'ancestor',
        wordnet_filename: str = "bansal14_trees.csv",
        wordnet_df_names: List[str] = ['hypernym', 'term', 'tree_id', 'train_test_split'],
        keys_to_print: List[str] = ['pruned_precision', 'pruned_recall', 'pruned_f1']):

    sentences, results, wordnet_df = get_sentences_results_wordnet_df(
            wordnet_filepath=os.path.join(wordnet_dir, wordnet_filename),
            results_filepath=os.path.join(results_dir, results_filename.format(epoch_num=epoch_num)),
            sentences_filepath=os.path.join(data_dir, test_filename))
    tree_ids = np.unique([val["tree_id"] for val in sentences.values()])

    subtrees_dict = {}
    subtrees_info_dict = {}

    for tree_id in tree_ids:
        run_inference_subtree(
            tree_id=tree_id,
            prediction_metric_type=prediction_metric_type,
            sentences=sentences,
            results=results,
            subtrees_dict=subtrees_dict,
            subtrees_info_dict=subtrees_info_dict,
            wordnet_df=wordnet_df,
        )

    print_average_metrics(subtrees_info_dict, epoch_num)

    with open(os.path.join(results_dir, subtrees_filename.format(epoch_num=epoch_num)), 'wb') as f:
        pickle.dump(subtrees_dict, f)


def run_validation(test_data,
        test_filename,
        train_filename,
        model,
        results_dir,
        epoch_num,
        experiment_name,
        results_filename,
        metrics_filename):
    """
    test_data: (test_data, test_hashes).
    """
    test_dataloader, test_hashes = test_data
    eval_accuracy = 0
    nb_eval_steps = 0
    results_dict = {}
    hash_index = 0  # TODO: Better way to associate hash with example?
    t0 = time.time()
    for batch_num, batch in enumerate(tqdm.tqdm(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(input_ids=b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask)[0]

        logits = outputs

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        for input_id, logit, label_id in zip(b_input_ids, logits, label_ids):
            example_id = test_hashes[hash_index]
            hash_index += 1
            results_dict[example_id] = {
                "logits": logit.tolist(),
                "label": [int(label_id)],
            }

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    print(f"Validation Accuracy: {round(eval_accuracy / nb_eval_steps, 2)}")
    print(f"Validation took: {format_time(time.time() - t0)}")

    with open(os.path.join(results_dir, results_filename.format(epoch_num=epoch_num)), "w") as f:
        json.dump(results_dict, f)

    metrics_dict = compute_scores(results_dict=results_dict)
    print("  metrics", metrics_dict)
    with open(os.path.join(results_dir, metrics_filename.format(epoch_num=epoch_num)), "w") as f:
        json.dump(metrics_dict, f)

    return eval_accuracy / nb_eval_steps


def train(
    train_filename,
    test_filenames,
    subtrees_filename,
    data_dir,
    results_dir,
    ckpt_dir,
    results_filename,
    ckpt_filename,
    metrics_filename,
    test_wordnet_filenames,
    batch_size,
    num_epochs,
    learning_rate,
    num_warmup_steps,
    num_labels,
    experiment_name,
    cased,
    reload_from_epoch_num,
    validate_only,
    model,
    reload_from_checkpoint,
    max_len,
    dataset_type='sequence_classification',
    adam_eps=1e-8,
    seed_val=2020,
):
    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForSequenceClassification.from_pretrained(model)

    if reload_from_checkpoint:
        ckpt_filename = os.path.join(ckpt_dir, ckpt_filename.format(epoch_num=reload_from_epoch_num))
        print(f"reloading from {ckpt_filename}")
        model.load_state_dict(torch.load(ckpt_filename))
        starting_epoch_num = int(reload_from_epoch_num.split("_")[0])
    else:
        starting_epoch_num = -1

    t0 = time.time()

    if not args.validate_only:
        print(f"{format_time(time.time()-t0)} Getting train dataset")
        train_dataloader, _ = get_dataset(train_filename,
                data_dir,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_len=max_len,
                dataset_type=dataset_type,
                is_train_data=True)
        num_training_steps = num_epochs * len(train_dataloader)
        first_epoch_validation_increment = len(train_dataloader) // 10
        optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_eps)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)

    print(f"{format_time(time.time()-t0)} Getting validation dataset")
    test_datas_list = [get_dataset(test_filename,
        data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        dataset_type=dataset_type,
        is_train_data=False)
        for test_filename in test_filenames]

    model.to(device)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    print(f"{format_time(time.time()-t0)} Begin training")

    def do_validation():
        print("")
        print("Running Validation...")
        for test_data, test_filename, test_wordnet_filename in zip(test_datas_list, test_filenames, test_wordnet_filenames):
            run_validation(test_data=test_data,
                    test_filename=test_filename,
                    train_filename=train_filename,
                    model=model,
                    results_dir=results_dir,
                    epoch_num=epoch_num,
                    experiment_name=experiment_name,
                    results_filename=results_filename,
                    metrics_filename=metrics_filename)

            run_inference(test_filename=test_filename,
                    epoch_num=epoch_num,
                    results_filename=results_filename,
                    subtrees_filename=subtrees_filename,
                    results_dir=results_dir,
                    wordnet_filename=test_wordnet_filename,
                    data_dir=data_dir)

    if validate_only:
        epoch_num = reload_from_epoch_num
        do_validation()
        return

    for epoch_num in range(starting_epoch_num + 1, starting_epoch_num + 1 + num_epochs):
        print(f"Epoch num: {epoch_num}")
        do_validation()
        t0 = time.time()
        total_loss = 0
        train_accuracy = 0
        nb_train_steps = 0
        model.train()
        for step, batch in enumerate(tqdm.tqdm(train_dataloader)):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step}  of  {len(train_dataloader)}.    Elapsed: {elapsed}.")

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()
            outputs = model(b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask)[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, num_labels), b_labels.view(-1))
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            logits = outputs
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_train_accuracy = flat_accuracy(logits, label_ids)
            train_accuracy += tmp_train_accuracy
            nb_train_steps += 1

            if (epoch_num == 0
                    and step % first_epoch_validation_increment == 0
                    and not step == 0):
                print(f"    Step {step}, epoch {epoch_num}")
                do_validation()
                torch.save(model.state_dict(),
                        os.path.join(ckpt_dir, ckpt_filename.format(epoch_num=epoch_num)).replace('.ckpt', f'_step_{step}.ckpt'))
                compute_metrics(labels=label_ids, label_predictions=np.argmax(logits, axis=-1))

        avg_train_loss = total_loss / len(train_dataloader)
        loss_values.append(avg_train_loss)

        print(f"  Training Accuracy: {round(train_accuracy / nb_train_steps, 2)}")
        print(f"  Average training loss: {round(avg_train_loss, 2)}")
        print(f"  Training epoch took: {format_time(time.time() - t0)}")
        torch.save(model.state_dict(),
            os.path.join(ckpt_dir, ckpt_filename.format(epoch_num=epoch_num)))
        compute_metrics(labels=label_ids, label_predictions=np.argmax(logits, axis=1))

        t0 = time.time()

        model.eval()
    print("")
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filename", type=str)
    parser.add_argument("--test-filenames",
            nargs="+",
            type=str,
            help='list of parts of test filenames before ".taxo" (e.g. WN_plants_test WN_vehicles_test)')
    parser.add_argument("--subtrees-filename", type=str)
    parser.add_argument("--data-dir", type=str, default="./datasets/generated_training_pairs")
    parser.add_argument("--results-dir", type=str, default="./outputs/results/")
    parser.add_argument("--ckpt-dir", type=str, default="./outputs/ckpts/")
    parser.add_argument("--results-filename", type=str)
    parser.add_argument("--metrics-filename", type=str)
    parser.add_argument("--test-wordnet-filenames", type=str, nargs='+')
    parser.add_argument("--ckpt-filename", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-epochs", default=10, type=int)
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--num-warmup-steps", default=320, type=int)
    parser.add_argument("--num-labels", default=2, type=int)
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument("--cased", type=str2bool)
    parser.add_argument("--reload-from-epoch-num", default=-1)
    parser.add_argument("--validate-only", type=str2bool, default="false")
    parser.add_argument("--reload-from-checkpoint", type=str2bool, default="false")
    parser.add_argument("--model", type=str, help="bert, roberta, etc")
    parser.add_argument("--max-len", default=64, type=int)
    parser.add_argument("--dataset-type", type=str, default='sequence_classification')
    parser.add_argument("--random-seed", type=int, default=0)

    args = parser.parse_args()
    check_backwards_compatability(args)
    print(f"args {args}")

    if not os.path.exists("./outputs/logs"):
        os.makedirs("./outputs/logs")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    print('Experiment running on commit ', subprocess.check_output(['git', 'rev-parse', 'HEAD']))

    set_random_seed(args.random_seed)
    train(train_filename=args.train_filename,
          test_filenames=args.test_filenames,
          subtrees_filename=args.subtrees_filename,
          data_dir=args.data_dir,
          results_dir=args.results_dir,
          ckpt_dir=args.ckpt_dir,
          results_filename=args.results_filename,
          ckpt_filename=args.ckpt_filename,
          metrics_filename=args.metrics_filename,
          batch_size=args.batch_size,
          num_epochs=args.num_epochs,
          learning_rate=args.learning_rate,
          num_warmup_steps=args.num_warmup_steps,
          num_labels=args.num_labels,
          cased=args.cased,
          experiment_name=args.experiment_name,
          reload_from_checkpoint=args.reload_from_checkpoint,
          reload_from_epoch_num=args.reload_from_epoch_num,
          validate_only=args.validate_only,
          model=args.model,
          max_len=args.max_len,
          dataset_type=args.dataset_type,
          test_wordnet_filenames=args.test_wordnet_filenames)
