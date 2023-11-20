# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script is adapted from the work by winners of the CNC 2022 Subtask 2 @ CASE, Team 1Cademy.
Original Repository: https://github.com/Gzhang-umich/1CademyTeamOfCASE
Main Script to Refer to: "run_st2_v2.py"

Script taken and adapted from https://github.com/tanfiona/CausalNewsCorpus/blob/master/run_st2.py, keeping the
copyright notices intact.
"""

import argparse
import json
import logging
import os
import random
from copy import deepcopy
from logging import Logger
from typing import Iterable, Tuple

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset as HF_Dataset
from datasets import concatenate_datasets, load_dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CONFIG_MAPPING, AutoConfig, AutoTokenizer

from evaluation.utils_eval_st2 import main as eval_st2
from models.bilou_tagger_st2 import NER_CRF_Classifier
from utils.bilou_tags import THREE_LAYER_BILOU_TAGS, tl_bilou_id2ne_label
from utils.cnc_dataset import CausalNewsDatasetST2, get_CES_bounds
from utils.constants import CPU, DEVICE
from utils.sqrt_scheduler import SqrtSchedule

args = None

logger = Logger(__name__)

ARG0_BEGIN: str = "<ARG0>"
ARG0_END: str = "</ARG0>"
ARG1_BEGIN: str = "<ARG1>"
ARG1_END: str = "</ARG1>"
SIG0_BEGIN: str = "<SIG0>"
SIG0_END: str = "</SIG0>"

tokenizer = None


def set_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def move_to_device(device: str, *tensors: Iterable[torch.Tensor]) -> Tuple[torch.Tensor]:
    """
    Moves a tuple of tensors to the specified memory space.

    Args:
        device (str): The memory space to move to (GPU or CPU)

    Returns:
        Tuple[torch.Tensor]: References to the moved tensors
    """
    return tuple([t.to(device) for t in tensors])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune BERT- and CRF-based model for span extraction."
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="A csv or a json file containing the test data.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use for the BERT-based LM.",
    )

    parser.add_argument(
        "--learning_rate_crf",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use for the CRF.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="classifier dropout rate",
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to train models from scratch.",
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to use model to predict on test set.",
    )
    parser.add_argument(
        "--augmentation_file",
        type=str,
        default=None,
        help="Whether to use additional augmented data.",
    )
    parser.add_argument("--id", type=int, default=1, help="ID of the run used for metrics cache.")
    parser.add_argument(
        "--clone_multi_layer_instances",
        action="store_true",
        help="Randomly clones multi-relation instances.",
    )
    parser.add_argument(
        "--clone_percentage",
        type=float,
        default=0.3,
        help="Amount of multi-relation instances to clone relative to entire dataset size.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None or args.validation_file is None or args.test_file is None:
        raise ValueError("Need training/validation/test files.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
            ], "`validation_file` should be a csv or a json file."
        if args.test_file is not None:
            extension = args.test_file.split(".")[-1]
            assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."

    return args


def get_final_sentences(
    model: NER_CRF_Classifier,
    prediction_labels: list[list[str]],
    eval_dataset: CausalNewsDatasetST2,
) -> list[list[str]]:
    final_sentences: list[list[str]] = []
    for k, pred_l in enumerate(prediction_labels):
        final_sents = []
        if len(eval_dataset.tokenized_inputs["tokens"][k]) == 0:
            final_sentences.append(["", "", ""])
            continue
        for y in range(3):
            pl = pred_l[y]
            arg0_begin = -1
            arg0_end = -1
            arg1_begin = -1
            arg1_end = -1
            sig0_begin = -1
            sig0_end = -1

            found: bool = False
            # Iterate over every entry
            # Search for ARG1
            for i in range(len(pl)):
                if "U-ARG1" in pl[i]:
                    arg1_begin = arg1_end = i
                    break
                elif "B-ARG1" in pl[i]:
                    arg1_begin = i
                    j = i + 1
                    while j < len(pl):
                        if "I-ARG1" in pl[j]:
                            j += 1
                            continue
                        elif "L-ARG1" in pl[j]:
                            arg1_end = j
                            found = True
                            break
                        else:
                            break
                    if not found:
                        arg1_begin = -1
                    break

            found = False

            for i in range(len(pl)):
                if "U-ARG0" in pl[i]:
                    arg0_begin = arg0_end = i
                elif "B-ARG0" in pl[i]:
                    arg0_begin = i
                    j = i + 1
                    while j < len(pl):
                        if "I-ARG0" in pl[j]:
                            j += 1
                            continue
                        elif "L-ARG0" in pl[j]:
                            arg0_end = j
                            found = True
                            break
                        else:
                            break
                    if not found:
                        arg0_begin = -1
                    break

            found = False

            for i in range(len(pl)):
                if "U-SIG0" in pl[i]:
                    sig0_begin = sig0_end = i
                elif "B-SIG0" in pl[i]:
                    sig0_begin = i
                    j = i + 1
                    while j < len(pl):
                        if "I-SIG0" in pl[j]:
                            j += 1
                            continue
                        elif "L-SIG0" in pl[j]:
                            sig0_end = j
                            found = True
                            break
                        else:
                            break
                    if not found:
                        sig0_begin = -1
                    break

            tokens: list[str] = deepcopy(eval_dataset.tokenized_inputs["tokens"][k])
            if len(tokens) == 0:
                final_sents.append("")
            if arg1_begin != -1 and arg1_end != -1:
                tokens[arg1_begin] = ARG1_BEGIN + tokens[arg1_begin]
                tokens[arg1_end] += ARG1_END
            if arg0_begin != -1 and arg0_end != -1:
                tokens[arg0_begin] = ARG0_BEGIN + tokens[arg0_begin]
                tokens[arg0_end] += ARG0_END
            if sig0_begin != -1 and sig0_end != -1:
                tokens[sig0_begin] = SIG0_BEGIN + tokens[sig0_begin]
                tokens[sig0_end] += SIG0_END
            if not all(tag == "O" for tag in pl):
                final_sents.append(" ".join(tokens))
        final_sentences.append(final_sents)

    return final_sentences


def main():
    """
    Entry point.

    Returns:
        int: Status code
    """
    global args
    args = parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
        extension = args.train_file.split(".")[-1]
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file
        extension = args.validation_file.split(".")[-1]
    if args.test_file is not None:
        data_files["test"] = args.test_file
        extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    if args.augmentation_file is not None:
        augment_dataset = HF_Dataset.from_pandas(pd.read_csv(args.augmentation_file))

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    global tokenizer

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True, add_prefix_space=True
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<ARG0>",
                "</ARG0>",
                "<ARG1>",
                "</ARG1>",
                "<SIG0>",
                "</SIG0>",
            ]
        }
    )

    def preprocessing(examples):
        all_tokens = []
        all_starts = []
        all_ends = []

        multi_layer_indices: list[int] = []

        for i, causal_text_w_pairs in enumerate(examples["causal_text_w_pairs"]):
            causal_text_w_pairs = eval(causal_text_w_pairs)
            example_tokens: list[str] = None
            example_starts: list[list[int]] = []
            example_ends: list[list[int]] = []

            if len(causal_text_w_pairs) > 0:
                if len(causal_text_w_pairs) > 1:
                    multi_layer_indices.append(len(all_tokens))
                for text in causal_text_w_pairs:
                    tokens, starts, ends = get_CES_bounds(text)
                    if example_tokens is None:
                        example_tokens = tokens
                    example_starts.append(starts)
                    example_ends.append(ends)
                all_tokens.append(example_tokens)
                all_starts.append(example_starts)
                all_ends.append(example_ends)

        if args.clone_multi_layer_instances:
            selected_clone_indices = np.random.choice(
                multi_layer_indices,
                size=int(args.clone_percentage * len(all_tokens)),
                replace=True,
            )
            for idx in selected_clone_indices:
                all_tokens.append(deepcopy(all_tokens[idx]))
                all_starts.append(deepcopy(all_starts[idx]))
                all_ends.append(deepcopy(all_ends[idx]))

        assert len(all_tokens) == len(all_starts)
        assert len(all_ends) == len(all_starts)
        return {"tokens": all_tokens, "all_starts": all_starts, "all_ends": all_ends}

    if args.train_file is not None:
        raw_datasets["train"] = raw_datasets["train"].map(
            preprocessing,
            batched=True,
            batch_size=len(raw_datasets["train"]),
            remove_columns=raw_datasets["train"].column_names,
        )

    if args.augmentation_file is not None and args.train_file is not None:
        augment_dataset = augment_dataset.map(
            preprocessing, batched=True, remove_columns=augment_dataset.column_names
        )
        raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], augment_dataset])

    if args.train_file is not None:
        train_dataset = CausalNewsDatasetST2(
            raw_datasets["train"], split="train", model_path=args.model_name_or_path
        )

    if args.validation_file is not None:
        truth = pd.read_csv(args.validation_file, sep=",", encoding="utf-8").reset_index(drop=True)
        eval_dataset = CausalNewsDatasetST2(
            raw_datasets["validation"], "validation", model_path=args.model_name_or_path
        )

    else:
        eval_dataset = None

    if args.test_file is not None:
        test_dataset = CausalNewsDatasetST2(
            raw_datasets["test"], "test", model_path=args.model_name_or_path
        )

    else:
        test_dataset = None

    model = NER_CRF_Classifier(args.model_name_or_path, THREE_LAYER_BILOU_TAGS)

    if train_dataset is not None:
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    else:
        train_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size)
    else:
        eval_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
    else:
        test_dataloader = None

    if args.do_train:
        lm_optimizer: torch.optim.AdamW = torch.optim.AdamW(
            model.get_lm_parameters(), lr=args.learning_rate
        )
        crf_optimizer: torch.optim.AdamW = torch.optim.AdamW(
            model.get_crf_parameters(), lr=args.learning_rate_crf
        )

        lm_scheduler = None
        crf_scheduler = None

        lm_scheduler: LambdaLR = LambdaLR(
            optimizer=lm_optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
        )
        crf_scheduler: LambdaLR = LambdaLR(
            optimizer=crf_optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
        )

        model.to(DEVICE)

        total_batch_size = args.batch_size * 1 * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        completed_steps = 0
        starting_epoch = 0

        best_f1: float = 0.0
        best_model = deepcopy(model)

        progress_bar = tqdm(range(len(train_dataloader)))

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            print(f"Starting epoch {epoch + 1}")
            progress_bar.reset()
            for step, batch in enumerate(train_dataloader):

                (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                ) = move_to_device(
                    DEVICE,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                )

                loss = -1 * model.get_loss(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                    batch["label_ids"],
                )
                loss.backward()

                (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                ) = move_to_device(
                    CPU,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                )

                loss = loss / args.gradient_accumulation_steps
                if (
                    step % args.gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    lm_optimizer.step()
                    crf_optimizer.step()
                    lm_optimizer.zero_grad()
                    crf_optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(f"Loss: {loss}")
                    completed_steps += 1

                lm_scheduler.step()
                crf_scheduler.step()

            model.eval()

            predictions = []
            final_sentences: list[list[str]] = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    (
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                        batch["crf_mask"],
                        batch["sorted_crf_mask"],
                    ) = move_to_device(
                        DEVICE,
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["token_type_ids"],
                        batch["crf_mask"],
                        batch["sorted_crf_mask"],
                    )

                pred = model.predict_tag_sequence(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                )

                (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                ) = move_to_device(
                    CPU,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                )

                predictions.extend(pred)

            prediction_labels: list[list[str]] = []
            for pred in predictions:
                prediction_labels.append([[], [], []])
                for p in pred:
                    layer1, layer2, layer3 = tl_bilou_id2ne_label[p].split("|")
                    prediction_labels[-1][0].append(layer1)
                    prediction_labels[-1][1].append(layer2)
                    prediction_labels[-1][2].append(layer3)

            final_sentences = get_final_sentences(model, prediction_labels, eval_dataset)

            scores = eval_st2(truth, final_sentences, id=args.id)

            if scores[0]["Overall"]["f1"] > best_f1:
                print(
                    f"Writing at epoch {epoch} with F1 score of {scores[0]['Overall']['f1']} and trad F1 {scores[0]['Overall']['trad_f1']}."
                )
                best_f1 = scores[0]["Overall"]["f1"]
                best_model = deepcopy(model)
                with open(
                    os.path.join(args.output_dir, "output_preds_valid.jsonl"), mode="w"
                ) as f:
                    for i in range(len(final_sentences)):
                        f.write(json.dumps({"index": i, "prediction": final_sentences[i]}) + "\n")

    if args.do_test:

        best_model = best_model.eval()
        predictions = []

        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                (
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                ) = move_to_device(
                    DEVICE,
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["token_type_ids"],
                    batch["crf_mask"],
                    batch["sorted_crf_mask"],
                )

            pred = best_model.predict_tag_sequence(
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            )

            (
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            ) = move_to_device(
                CPU,
                batch["input_ids"],
                batch["attention_mask"],
                batch["token_type_ids"],
                batch["crf_mask"],
                batch["sorted_crf_mask"],
            )
            predictions.extend(pred)

        prediction_labels: list[list[str]] = []
        for pred in predictions:
            prediction_labels.append([[], [], []])
            for p in pred:
                layer1, layer2, layer3 = tl_bilou_id2ne_label[p].split("|")
                prediction_labels[-1][0].append(layer1)
                prediction_labels[-1][1].append(layer2)
                prediction_labels[-1][2].append(layer3)

        final_sentences = get_final_sentences(model, prediction_labels, test_dataset)

        with open(os.path.join(args.output_dir, "output_preds_test.jsonl"), mode="w") as f:
            for i in range(len(final_sentences)):
                f.write(json.dumps({"index": i, "prediction": final_sentences[i]}) + "\n")

        return 0


if __name__ == "__main__":
    main()
