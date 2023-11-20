# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains dataset classes for dealing with the CNC corpus.
"""

import re

import numpy as np
import torch
from datasets.arrow_dataset import Dataset as ArrowDataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.bilou_tags import bilou_ne_label2id, tl_bilou_ne_label2id


class CausalNewsDatasetST1:
    """
    Dataset class for Subtask 1.
    """

    def __init__(self, dataset: ArrowDataset, model_path: str) -> None:
        """
        Loads and prepares the dataset for Pytorch-based training.

        Args:
            dataset (ArrowDataset): CNC dataset input
            model_path (str): Path (or name) of the BERT-based model used for the tokenizer.
        """
        if "roberta" in model_path:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                model_path, fast=True, add_prefix_space=True
            )
        else:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
        self._tokenized_input: torch.Tensor = tokenizer.batch_encode_plus(
            dataset["text"], padding=True, return_tensors="pt", pad_to_multiple_of=512
        )
        self._labels = dataset["label"]
        self._tokenized_input["label"] = torch.tensor(self._labels, dtype=torch.float)

    def __len__(self) -> int:
        """
        Returns length of the dataset, which is used by the Pytorch Dataloader iterator.

        Returns:
            int: Length
        """
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns an element from the dataset.

        Args:
            idx (int): Index to a specific element.

        Returns:
            dict: Input IDs, attention mask and corresponding label.
        """
        return {
            "input_ids": self._tokenized_input["input_ids"][idx],
            "attention_masks": self._tokenized_input["attention_mask"][idx],
            "label": self._tokenized_input["label"][idx],
        }


class CausalNewsTestDatasetST1:
    """
    Dataset class tailored to the test split of the CNC subtask 1 corpus.
    """

    def __init__(self, dataset: ArrowDataset, model_path: str) -> None:
        """
        Loads and prepares the dataset for Pytorch-based training.

        Args:
            dataset (ArrowDataset): CNC dataset input
            model_path (str): Path (or name) of the BERT-based model used for the tokenizer
        """
        if "roberta" in model_path:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                model_path, fast=True, add_prefix_space=True
            )
        else:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
        self._tokenized_input: torch.Tensor = tokenizer.batch_encode_plus(
            dataset["text"], padding=True, return_tensors="pt", pad_to_multiple_of=512
        )

    def __len__(self) -> int:
        """
        Returns length of the dataset, which is used by the Pytorch Dataloader iterator.

        Returns:
            int: Length
        """
        return len(self._tokenized_input["input_ids"])

    def __getitem__(self, idx) -> dict:
        """
        Returns an element from the dataset.

        Args:
            idx (int): Index to a specific element.

        Returns:
            dict: Input IDs, attention mask and corresponding label.
        """
        return {
            "input_ids": self._tokenized_input["input_ids"][idx],
            "attention_masks": self._tokenized_input["attention_mask"][idx],
        }


class CausalNewsDatasetST2(Dataset):
    """
    Dataset class used for Pytorch-based training in Subtask 2.
    """

    def __init__(self, examples: ArrowDataset, split: str, model_path: str):
        """
        Loads and prepared the CNC corpus for subtask 2.

        Args:
            examples (ArrowDataset): Samples
            split (str): Desired data split
            model_path (str): Path (or name) of the BERT-based model used for the tokenizer
        """
        if "roberta" in model_path:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
                model_path, fast=True, add_prefix_space=True
            )
        else:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path, fast=True)
        assert split in [
            "train",
            "validation",
            "test",
        ], "Split must be one of: train, validation, test"
        self.split = split
        self.indices: list[int] = []

        if split == "validation":

            all_tokens = [text.split() for text in examples["text"]]

            self.tokenized_inputs = tokenizer(
                [text.split() for text in examples["text"]],
                max_length=512,
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
            )

            all_tokens = []
            all_starts = []
            all_ends = []
            for i, causal_text_w_pairs in enumerate(examples["causal_text_w_pairs"]):
                causal_text_w_pairs = eval(causal_text_w_pairs)
                example_tokens: list[str] = None
                example_starts: list[list[int]] = []
                example_ends: list[list[int]] = []
                if len(causal_text_w_pairs) > 0:
                    for text in causal_text_w_pairs:
                        tokens, starts, ends = get_CES_bounds(text)
                        if example_tokens is None:
                            example_tokens = tokens
                        example_starts.append(starts)
                        example_ends.append(ends)
                    all_tokens.append(example_tokens)
                    all_starts.append(example_starts)
                    all_ends.append(example_ends)
                else:
                    all_tokens.append([])
                    all_starts.append([])
                    all_ends.append([])
            assert len(all_tokens) == len(all_starts)
            assert len(all_ends) == len(all_starts)

            self.tokenized_inputs["tokens"] = all_tokens
            self.tokenized_inputs["all_starts"] = all_starts
            self.tokenized_inputs["all_ends"] = all_ends

            if "token_type_ids" not in self.tokenized_inputs.data.keys():
                self.tokenized_inputs["token_type_ids"] = np.zeros(
                    shape=(
                        len(self.tokenized_inputs["input_ids"]),
                        len(self.tokenized_inputs["input_ids"][0]),
                    ),
                    dtype=int,
                ).tolist()
        elif split == "train":
            self.tokenized_inputs = tokenizer(
                examples["tokens"],
                max_length=512,
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
            )
            self.tokenized_inputs["tokens"] = examples["tokens"]
            self.tokenized_inputs["all_starts"] = examples["all_starts"]
            self.tokenized_inputs["all_ends"] = examples["all_ends"]

            if "token_type_ids" not in self.tokenized_inputs.data.keys():
                self.tokenized_inputs["token_type_ids"] = np.zeros(
                    shape=(
                        len(self.tokenized_inputs["input_ids"]),
                        len(self.tokenized_inputs["input_ids"][0]),
                    ),
                    dtype=int,
                ).tolist()

        else:
            all_tokens = [text.split() for text in examples["text"]]

            self.tokenized_inputs = tokenizer(
                [text.split() for text in examples["text"]],
                max_length=512,
                padding="max_length",
                truncation=True,
                is_split_into_words=True,
            )

            if "token_type_ids" not in self.tokenized_inputs.data.keys():
                self.tokenized_inputs["token_type_ids"] = np.zeros(
                    shape=(
                        len(self.tokenized_inputs["input_ids"]),
                        len(self.tokenized_inputs["input_ids"][0]),
                    ),
                    dtype=int,
                ).tolist()

            self.tokenized_inputs["tokens"] = all_tokens

            crf_masks: list[list[int]] = []

            for i, tokens in enumerate(self.tokenized_inputs["tokens"]):

                word_ids = self.tokenized_inputs.word_ids(batch_index=i)
                word2tok = {w: [] for w in range(len(tokens))}
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is not None:
                        word2tok[word_idx].append(token_idx)

                crf_mask: list[int] = [0] * 512

                # For all non-wordpiece tokens, add 1 in the CRF mask
                # CRF mask includes [CLS] and [SEP] token, so we need
                # to offset by 1

                for j in range(1, len(word2tok) + 1):
                    if len(word2tok[j - 1]) == 0:
                        continue
                    crf_mask[word2tok[j - 1][0]] = 1

                crf_masks.append(crf_mask)

            self.tokenized_inputs["crf_masks"] = crf_masks

            return

        # convert word to token tags
        converted_starts: list[list[int]] = []
        converted_ends: list[list[int]] = []
        store_word_ids = []
        all_bilou_labels: list[list[str]] = []
        all_bilou_ids: list[list[int]] = []
        crf_masks: list[list[int]] = []

        for i, tokens in enumerate(self.tokenized_inputs["tokens"]):

            if len(self.tokenized_inputs["all_starts"][i]) == 0:
                converted_starts.append([])
                converted_ends.append([])
                store_word_ids.append([])
                all_bilou_labels.append(["O"] * 512)
                all_bilou_ids.append([bilou_ne_label2id["O"]] * 512)
                crf_masks.append([0] * 512)
                continue

            word_ids = self.tokenized_inputs.word_ids(batch_index=i)
            word2tok = {w: [] for w in range(len(tokens))}
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is not None:
                    word2tok[word_idx].append(token_idx)

            current_starts: list[list[int]] = []
            current_ends: list[list[int]] = []

            for a_s in self.tokenized_inputs["all_starts"][i]:
                starts = []
                for a in a_s:
                    # sometimes, word2tok can be like
                    # {0: [1, 2], 1: [3], 2: [4], 3: [5, 6], 4: [7], 5: [8], 6: [9], 7: [10], 8: [], 9: [11, 12], 10: [], 11: [13], 12: [14], 13: [15], 14: [16, 17]}
                    # i.e. word --> empty token ids
                    # if so, we move to earlier word for starts
                    while (len(word2tok[int(a)]) == 0) and a >= 0:
                        a -= 1
                    starts.append(word2tok[int(a)][0])
                # for a in  self.tokenized_inputs["all_ends"][i]:
                if len(starts) <= 2:
                    # if missing signal, we put a dummy to ignore
                    starts.append(-100)
                current_starts.append(starts[:3])
            for a_e in self.tokenized_inputs["all_ends"][i]:
                ends = []
                for a in a_e:
                    while (len(word2tok[int(a)]) == 0) and a >= 0:
                        a += 1
                    ends.append(word2tok[int(a)][-1])

                # our code only predicts 1 signal for now
                if len(ends) <= 2:
                    # if missing signal, we put a dummy to ignore
                    ends.append(-100)
                current_ends.append(ends[:3])

            converted_starts.append(current_starts)
            converted_ends.append(current_ends)
            store_word_ids.append(word_ids)

            # Creating Nested Named Entity BIO Labels
            B: str = "B-{0}"
            I: str = "I-{0}"
            L: str = "L-{0}"
            O: str = "O"
            U: str = "U-{0}"

            crf_mask: list[int] = [0] * 512

            # For all non-wordpiece tokens, add 1 in the CRF mask
            # CRF mask includes [CLS] and [SEP] token, so we need
            # to offset by 1

            for j in range(1, len(word2tok) + 1):
                if len(word2tok[j - 1]) == 0:
                    continue
                crf_mask[word2tok[j - 1][0]] = 1

            bilou_labels_per_sample: list[list[str]] = []
            bilou_ids_per_sample: list[list[int]] = []

            # Start iterating over the all possible starts and ends and therefore create combined labels

            for j in range(3):

                bilou_labels: list[str] = [O] * len(word2tok)
                bilou_ids: list[int] = [bilou_ne_label2id[O]] * len(word2tok)

                if len(converted_starts[i]) <= j:
                    bilou_labels_per_sample.append(bilou_labels)
                    bilou_ids_per_sample.append(
                        bilou_ids + [bilou_ne_label2id["x"]] * (512 - len(bilou_ids))
                    )
                    continue

                # Start with ARG1 since it is the easier case

                # Find word index of start token
                start_word_index_arg1: int = [
                    k
                    for k, w in enumerate(word2tok.values())
                    if len(w) > 0 and w[0] == converted_starts[i][j][1]
                ][0]
                end_word_index_arg1: int = [
                    k
                    for k, w in enumerate(word2tok.values())
                    if len(w) > 0 and w[-1] == converted_ends[i][j][1]
                ][0]

                start_word_index_arg0: int = [
                    k
                    for k, w in enumerate(word2tok.values())
                    if len(w) > 0 and w[0] == converted_starts[i][j][0]
                ][0]
                end_word_index_arg0: int = [
                    k
                    for k, w in enumerate(word2tok.values())
                    if len(w) > 0 and w[-1] == converted_ends[i][j][0]
                ][0]

                start_word_index_sig0: int = -100
                if converted_starts[i][j][2] != -100:
                    start_word_index_sig0 = [
                        k
                        for k, w in enumerate(word2tok.values())
                        if len(w) > 0 and w[0] == converted_starts[i][j][2]
                    ][0]

                end_word_index_sig0: int = -100
                if converted_ends[i][j][2] != -100:
                    end_word_index_sig0: int = [
                        k
                        for k, w in enumerate(word2tok.values())
                        if len(w) > 0 and w[-1] == converted_ends[i][j][2]
                    ][0]

                # ARG 1
                if start_word_index_arg1 == end_word_index_arg1:
                    bilou_labels[start_word_index_arg1] = U.format("ARG1")
                    bilou_ids[start_word_index_arg1] = bilou_ne_label2id[U.format("ARG1")]
                else:
                    bilou_labels[start_word_index_arg1] = B.format("ARG1")
                    bilou_ids[start_word_index_arg1] = bilou_ne_label2id[B.format("ARG1")]

                    for k in range(start_word_index_arg1 + 1, end_word_index_arg1):
                        if len(word2tok[k]) == 0:
                            continue
                        bilou_labels[k] = I.format("ARG1")
                        bilou_ids[k] = bilou_ne_label2id[I.format("ARG1")]

                    bilou_labels[end_word_index_arg1] = L.format("ARG1")
                    bilou_ids[end_word_index_arg1] = bilou_ne_label2id[L.format("ARG1")]

                # ARG0
                if start_word_index_arg0 == end_word_index_arg0:
                    bilou_labels[start_word_index_arg0] = U.format("ARG0")
                    bilou_ids[start_word_index_arg0] = bilou_ne_label2id[U.format("ARG0")]
                else:
                    bilou_labels[start_word_index_arg0] = B.format("ARG0")
                    bilou_ids[start_word_index_arg0] = bilou_ne_label2id[B.format("ARG0")]
                    for k in range(start_word_index_arg0 + 1, end_word_index_arg0):
                        if len(word2tok[k]) == 0:
                            continue
                        bilou_labels[k] = I.format("ARG0")
                        bilou_ids[k] = bilou_ne_label2id[I.format("ARG0")]
                    bilou_labels[end_word_index_arg0] = L.format("ARG0")
                    bilou_ids[end_word_index_arg0] = bilou_ne_label2id[L.format("ARG0")]

                # SIG0
                if start_word_index_sig0 != -100 and end_word_index_sig0 != -100:
                    if start_word_index_sig0 == end_word_index_sig0:
                        if bilou_labels[start_word_index_sig0] == O:
                            bilou_labels[start_word_index_sig0] = U.format("SIG0")
                            bilou_ids[start_word_index_sig0] = bilou_ne_label2id[U.format("SIG0")]
                        else:
                            bilou_labels[start_word_index_sig0] += "+" + U.format("SIG0")
                            bilou_ids[start_word_index_sig0] = bilou_ne_label2id[
                                bilou_labels[start_word_index_sig0]
                            ]

                    else:
                        if bilou_labels[start_word_index_sig0] == O:
                            bilou_labels[start_word_index_sig0] = B.format("SIG0")
                            bilou_ids[start_word_index_sig0] = bilou_ne_label2id[B.format("SIG0")]
                        else:
                            bilou_labels[start_word_index_sig0] += "+" + B.format("SIG0")
                            bilou_ids[start_word_index_sig0] = bilou_ne_label2id[
                                bilou_labels[start_word_index_sig0]
                            ]

                        for k in range(start_word_index_sig0 + 1, end_word_index_sig0):
                            if len(word2tok[k]) == 0:
                                continue

                            if bilou_labels[k] == O:
                                bilou_labels[k] = I.format("SIG0")
                                bilou_ids[k] = bilou_ne_label2id[I.format("SIG0")]
                            else:
                                bilou_labels[k] += "+" + I.format("SIG0")
                                bilou_ids[k] = bilou_ne_label2id[bilou_labels[k]]

                        if bilou_labels[end_word_index_sig0] == O:
                            bilou_labels[end_word_index_sig0] = L.format("SIG0")
                            bilou_ids[end_word_index_sig0] = bilou_ne_label2id[L.format("SIG0")]
                        else:
                            bilou_labels[end_word_index_sig0] += "+" + L.format("SIG0")
                            bilou_ids[end_word_index_sig0] = bilou_ne_label2id[
                                bilou_labels[end_word_index_sig0]
                            ]

                bilou_labels_per_sample.append(bilou_labels)
                bilou_ids_per_sample.append(
                    bilou_ids + [bilou_ne_label2id["x"]] * (512 - len(bilou_ids))
                )

            # if len(bilou_labels_per_sample) == 2:
            #    all_bilou_labels.append(["{0}|{1}".format(l1, l2) for l1, l2 in zip(bilou_labels_per_sample[0], bilou_labels_per_sample[1])])
            #    all_bilou_ids.append([tl_bilou_ne_label2id[l] for l in all_bilou_labels[-1]] + [tl_bilou_ne_label2id["x"]] * (512 - len(all_bilou_labels[-1])))
            # elif len(bilou_labels_per_sample) == 3:
            all_bilou_labels.append(
                [
                    "{0}|{1}|{2}".format(l1, l2, l3)
                    for l1, l2, l3 in zip(
                        bilou_labels_per_sample[0],
                        bilou_labels_per_sample[1],
                        bilou_labels_per_sample[2],
                    )
                ]
            )
            all_bilou_ids.append(
                [tl_bilou_ne_label2id[tag] for tag in all_bilou_labels[-1]]
                + [tl_bilou_ne_label2id["x"]] * (512 - len(all_bilou_labels[-1]))
            )
            # else:
            #    all_bilou_labels.append(bilou_labels_per_sample[0])
            #    all_bilou_ids.append(bilou_ids_per_sample[0])
            crf_masks.append(crf_mask)

        self.tokenized_inputs["start_positions"] = converted_starts
        self.tokenized_inputs["end_positions"] = converted_ends
        self.tokenized_inputs["bilou_labels"] = all_bilou_labels
        self.tokenized_inputs["bilou_ids"] = all_bilou_ids
        self.tokenized_inputs["crf_masks"] = crf_masks
        # tokenized_inputs["store_word_ids"] = store_word_ids

    def __len__(self) -> int:
        """
        Returns length of the dataset.

        Returns:
            int: Length
        """
        return len(self.tokenized_inputs["input_ids"])

    def __getitem__(self, index: int) -> dict:
        """
        Returns a specific element in the dataset.

        Args:
            index (int): Index to the element

        Returns:
            dict: Input IDs, attention mask, token type IDs, CRF mask and sorted CRF mask
        """
        if self.split == "test":
            output_batch: dict = {
                "input_ids": torch.tensor(self.tokenized_inputs["input_ids"][index]),
                "attention_mask": torch.tensor(self.tokenized_inputs["attention_mask"][index]),
                "token_type_ids": torch.tensor(self.tokenized_inputs["token_type_ids"][index]),
                "crf_mask": torch.tensor(self.tokenized_inputs["crf_masks"][index]),
                "sorted_crf_mask": torch.sort(
                    torch.tensor(self.tokenized_inputs["crf_masks"][index]),
                    dim=0,
                    descending=True,
                    stable=True,
                )[0],
            }
        else:
            output_batch: dict = {
                "input_ids": torch.tensor(self.tokenized_inputs["input_ids"][index]),
                "attention_mask": torch.tensor(self.tokenized_inputs["attention_mask"][index]),
                "token_type_ids": torch.tensor(self.tokenized_inputs["token_type_ids"][index]),
                "label_ids": torch.tensor(self.tokenized_inputs["bilou_ids"][index]),
                "crf_mask": torch.tensor(self.tokenized_inputs["crf_masks"][index]),
                "sorted_crf_mask": torch.sort(
                    torch.tensor(self.tokenized_inputs["crf_masks"][index]),
                    dim=0,
                    descending=True,
                    stable=True,
                )[0],
            }
        return output_batch


# The functions below were taken from https://github.com/tanfiona/CausalNewsCorpus/blob/master/run_st2.py


def clean_tok(tok):
    # Remove all other tags: E.g. <SIG0>, <SIG1>...
    return re.sub("</*[A-Z]+\d*>", "", tok)  # noqa: W605


def get_CE_bounds(text_w_pairs):
    tokens = []
    cause = []
    effect = []

    for i, tok in enumerate(text_w_pairs.split(" ")):

        # Replace if special
        if "<ARG0>" in tok:
            tok = re.sub("<ARG0>", "", tok)
            cause.append(i)
        if "</ARG0>" in tok:
            tok = re.sub("</ARG0>", "", tok)
            cause.append(i)
        if "<ARG1>" in tok:
            tok = re.sub("<ARG1>", "", tok)
            effect.append(i)
        if "</ARG1>" in tok:
            tok = re.sub("</ARG1>", "", tok)
            effect.append(i)
        tokens.append(clean_tok(tok))

    start_positions = [cause[0], effect[0]]
    end_positions = [cause[1], effect[1]]

    return tokens, start_positions, end_positions


def get_S_bounds(text_w_pairs):
    tokens = []
    start_positions = []
    end_positions = []

    for i, tok in enumerate(text_w_pairs.split(" ")):
        # Replace if special
        if "<SIG" in tok:
            tok = re.sub("<SIG([A-Z]|\d)*>", "", tok)  # noqa: W605
            start_positions.append(i)

            if "</SIG" in tok:  # one word only
                tok = re.sub("</SIG([A-Z]|\d)*>", "", tok)  # noqa: W605
                end_positions.append(i)

        elif "</SIG" in tok:
            tok = re.sub("</SIG([A-Z]|\d)*>", "", tok)  # noqa: W605
            end_positions.append(i)

        tokens.append(clean_tok(tok))

    # workaround for errors where there are no closing bounds for SIG
    min_len = min(len(start_positions), len(end_positions))

    return tokens, start_positions[:min_len], end_positions[:min_len]


def get_CES_bounds(text_w_pairs):
    tokens, starts, ends = get_CE_bounds(text_w_pairs)
    tokens_s, starts_s, ends_s = get_S_bounds(text_w_pairs)
    assert tokens == tokens_s
    assert len(starts) == len(ends)
    assert len(starts_s) == len(ends_s)
    return tokens, starts + starts_s, ends + ends_s
