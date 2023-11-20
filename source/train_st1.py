# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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

# Script adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner.py

"""
This module runs the training pipeline for Subtask 1.
"""

import json
import logging
import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import Features, Value, load_dataset
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed

from models.binary_classifier_st1 import BinaryClassifier
from utils.cnc_dataset import CausalNewsDatasetST1, CausalNewsTestDatasetST1
from utils.constants import DEVICE
from utils.sqrt_scheduler import SqrtSchedule

logger = logging.getLogger(__name__)

features: dict[str, Value] = {
    "index": Value("string"),
    "text": Value("string"),
    "label": Value("int64"),
}

features_test: dict[str, Value] = {
    "index": Value("string"),
    "text": Value("string"),
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    use_weighted_ce: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If enabled, a weighted cross entropy loss on two output neurons will be used"
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default="roberta-base",
        metadata={"help": "Pretrained config name or path of BERT-based model."},
    )


def main() -> int:
    """
    Entry point.

    Returns:
        int: Status code
    """

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for arg in vars(training_args):
        print(arg, getattr(training_args, arg))

    for arg in vars(model_args):
        print(arg, getattr(model_args, arg))

    for arg in vars(data_args):
        print(arg, getattr(data_args, arg))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_files = {}
    if training_args.do_train:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    if (training_args.do_train and data_args.train_file.endswith(".csv")) or (
        training_args.do_predict and data_args.test_file.endswith(".csv")
    ):
        # Loading a dataset from local csv files
        raw_datasets = load_dataset(
            "csv",
            data_files=data_files,
            features=Features(features),
        )
    else:
        # Loading a dataset from local json files
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            features=Features(features),
        )

    test_datasets = load_dataset(
        "csv", data_files=[data_args.test_file], features=Features(features_test)
    )

    model = BinaryClassifier(model_args.model_name_or_path, data_args.use_weighted_ce)

    train_dataset: CausalNewsDatasetST1 = CausalNewsDatasetST1(
        raw_datasets["train"], model_args.model_name_or_path
    )
    dev_dataset: CausalNewsDatasetST1 = CausalNewsDatasetST1(
        raw_datasets["validation"], model_args.model_name_or_path
    )
    test_dataset: CausalNewsTestDatasetST1 = CausalNewsTestDatasetST1(
        test_datasets["train"], model_args.model_name_or_path
    )  # Yes, thats correct

    train_dataloader: DataLoader = DataLoader(
        train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True
    )
    dev_dataloader: DataLoader = DataLoader(
        dev_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False
    )
    test_dataloader: DataLoader = DataLoader(
        test_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=False
    )

    optimizer: AdamW = AdamW(model.parameters(), lr=training_args.learning_rate)
    scheduler: LambdaLR = LambdaLR(
        optimizer=optimizer, lr_lambda=SqrtSchedule(len(train_dataloader))
    )

    model = model.to(DEVICE)
    weight_tensor: torch.Tensor = torch.tensor([1.0, 1.5]).to(DEVICE)

    best_f1: float = 0.0
    best_model = None
    early_stopping_counter: int = 0

    for i in range(int(training_args.num_train_epochs)):
        model = model.train()
        with tqdm(len(train_dataloader)) as progress_bar:
            for batch in tqdm(train_dataloader):
                batch["input_ids"] = batch["input_ids"].to(DEVICE)
                batch["attention_masks"] = batch["attention_masks"].to(DEVICE)
                batch["label"] = batch["label"].to(DEVICE)
                preds = model(batch["input_ids"], batch["attention_masks"])
                if data_args.use_weighted_ce:
                    loss = cross_entropy(preds, batch["label"].long(), weight_tensor)
                else:
                    loss = binary_cross_entropy(preds.flatten(), batch["label"])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                batch["input_ids"] = batch["input_ids"].to("cpu")
                batch["attention_masks"] = batch["attention_masks"].to("cpu")
                batch["label"] = batch["label"].to("cpu")
                progress_bar.update(1)
                progress_bar.set_description(f"Loss: {float(loss)}")

        # Evaluation
        model = model.eval()
        eval_preds: list[int] = []
        print("Evaluating")
        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                batch["input_ids"] = batch["input_ids"].to(DEVICE)
                batch["attention_masks"] = batch["attention_masks"].to(DEVICE)
                preds = model(batch["input_ids"], batch["attention_masks"])
                if data_args.use_weighted_ce:
                    eval_preds.extend(torch.argmax(preds, dim=1).int().tolist())
                else:
                    eval_preds.extend((preds.flatten() > 0.5).int().tolist())
                batch["input_ids"] = batch["input_ids"].to("cpu")
                batch["attention_masks"] = batch["attention_masks"].to("cpu")

        prec, rec, f1, s = precision_recall_fscore_support(dev_dataset._labels, eval_preds)

        print(f"Epoch {i}: Prec. {prec}, Rec. {rec}, F1 {f1}")

        avg_f1 = (f1[0] + f1[1]) / 2

        if avg_f1 > best_f1:
            early_stopping_counter = 0
            best_f1 = avg_f1
            best_model = deepcopy(model)
            print(f"Writing at epoch {i} with F1 score of {avg_f1}.")
            with open(
                os.path.join(training_args.output_dir, "output_preds_valid.jsonl"), mode="w"
            ) as f:
                for x, p in enumerate(eval_preds):
                    f.write(json.dumps({"index": x, "prediction": p}) + "\n")
        else:
            early_stopping_counter += 1

        if early_stopping_counter == 3:
            print(f"Early stopping in epoch {i}")
            break

    best_model = best_model.eval()
    test_preds: list[int] = []
    print("Testing")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch["input_ids"] = batch["input_ids"].to(DEVICE)
            batch["attention_masks"] = batch["attention_masks"].to(DEVICE)
            preds = best_model(batch["input_ids"], batch["attention_masks"])
            if data_args.use_weighted_ce:
                test_preds.extend(torch.argmax(preds, dim=1).int().tolist())
            else:
                test_preds.extend((preds.flatten() > 0.5).int().tolist())
            batch["input_ids"] = batch["input_ids"].to("cpu")
            batch["attention_masks"] = batch["attention_masks"].to("cpu")

    with open(os.path.join(training_args.output_dir, "output_preds_test.jsonl"), mode="w") as f:
        for x, p in enumerate(test_preds):
            f.write(json.dumps({"index": x, "prediction": p}) + "\n")

    return 0


if __name__ == "__main__":
    main()
