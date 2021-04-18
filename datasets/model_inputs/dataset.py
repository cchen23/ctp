import os
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


def get_dataset(
    examples_filename,
    data_dir,
    tokenizer,
    batch_size,
    max_len,
    dataset_type=None,
    is_train_data=False
):
    """
    Returns TensorDataset constructed from text file of examples.

    """

    labels = []
    with open(os.path.join(data_dir, f"{examples_filename}"), "r") as f:
        data_dict = json.load(f)

    labels = [example["label"] for example in data_dict.values()]
    hashes = list(data_dict.keys())
    if dataset_type == 'wordnet_definition':
        if hasattr(tokenizer, 'encode_plus'):
            encoded_sentences = [
                    tokenizer.encode_plus(f'{example["term"].replace("_", " ")} {example["term_definition"]}', f'{example["hypernym"].replace("_", " ")} {example["hypernym_definition"]}', pad_to_max_length=True, max_length=max_len) for example in data_dict.values()]
        else:
            encoded_sentences = [
                    tokenizer.encode(f'{example["term"].replace("_", " ")} {example["term_definition"]}', f'{example["hypernym"].replace("_", " ")} {example["hypernym_definition"]}', add_special_tokens=True) for example in data_dict.values()
            ]
            for encoded_sentence in encoded_sentences:
                encoded_sentence.pad(max_len)
    elif dataset_type == 'retrieval_backoff':
        print('retrieval_backoff')
        if hasattr(tokenizer, 'encode_plus'):
            encoded_sentences = [
                    tokenizer.encode_plus(f'{example["term"].replace("_", " ")} <d> {example["term_context"]}', f'{example["hypernym"].replace("_", " ")} <d> {example["hypernym_context"]}', pad_to_max_length=True, max_length=max_len) for example in data_dict.values()]
        else:
            encoded_sentences = [
                    tokenizer.encode(f'{example["term"].replace("_", " ")} <d> {example["term_context"]}', f'{example["hypernym"].replace("_", " ")} <d> {example["hypernym_context"]}', add_special_tokens=True) for example in data_dict.values()]
            for encoded_sentence in encoded_sentences:
                encoded_sentence.pad(max_len)
        print('first 10 sentences:', '\n'.join([str(sentence) for sentence in encoded_sentences[:10]]))

    else:
        sentences = [example["sentence"] for example in data_dict.values()]
        encoded_sentences = [
            tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences
        ]
        if hasattr(tokenizer, 'encode_plus'):
            encoded_sentences = [tokenizer.encode_plus(sentence, pad_to_max_length=True, max_length=max_len) for sentence in sentences]
        else:
            encoded_sentences = [tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]
            for encoded_sentence in encoded_sentences:
                encoded_sentence.pad(max_len)
    print(f"Using {len(encoded_sentences)} sentences from {examples_filename}")
    try:
        inputs = torch.tensor(
            [encoded_sentence.input_ids for encoded_sentence in encoded_sentences]
        )
    except Exception:
        inputs = torch.tensor(
            [encoded_sentence.ids for encoded_sentence in encoded_sentences]
        )
    prediction_masks = torch.tensor(
        [encoded_sentence.attention_mask for encoded_sentence in encoded_sentences]
    )

    labels = torch.tensor(labels)
    data = TensorDataset(inputs, prediction_masks, labels)
    if is_train_data:
        sampler = RandomSampler(data)
        dataloader = DataLoader(
            data, sampler=sampler, batch_size=batch_size, num_workers=8
        )
    else:
        dataloader = DataLoader(
            data, shuffle=False, batch_size=batch_size, num_workers=8
        )
    print(f'{len(dataloader)} examples loaded from {examples_filename}')
    return dataloader, hashes
