#!/bin/bash

# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo -e '\033[1;31mExecute this script from the directory containing it! \033[0m'
PROJECT_ROOT=$(realpath "..")

# Set training parameters here
dropout=0.1
lr="7e-5"
crfLr="3e-4"
numEpochs=30
batchSize=16
modelNameOrPath="roberta-large"
seed=40637
outputDir="../output/st2"

# Select Training File
# If EDA augmented data is available, select this one instead of the original one

# trainingFile=$PROJECT_ROOT/augmented_data/eda_train_subtask2_grouped.csv
trainingFile=$PROJECT_ROOT/CausalNewsCorpus/data/V2/train_subtask2_grouped.csv

python $PROJECT_ROOT/source/train_st2.py                                            \
--learning_rate $lr                                                                 \
--learning_rate_crf $crfLr                                                          \
--train_file $trainingFile                                                          \
--do_train                                                                          \
--validation_file $PROJECT_ROOT/CausalNewsCorpus/data/V2/dev_subtask2_grouped.csv   \
--test_file $PROJECT_ROOT/CausalNewsCorpus/data/V2/test_subtask2_text.csv           \
--do_test                                                                           \
--dropout $dropout                                                                  \
--model_name_or_path $modelNameOrPath                                               \
--num_train_epochs $numEpochs                                                       \
--output_dir $outputDir                                                             \
--batch_size $batchSize                                                             \
--seed $seed
