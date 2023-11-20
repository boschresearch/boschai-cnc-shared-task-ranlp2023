#!/bin/bash

# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

echo -e '\033[1;31mExecute this script from the directory containing it! \033[0m'
PROJECT_ROOT=$(realpath "..")

# Set training parameters here
lr="8e-6"
seed=7885
numEpochs=40
batchSize=32
outputDir="../output/st1"
modelNameOrPath="roberta-large"
useWeightedCE=true

# Select Training File
# If EDA augmented data is available, select this one instead of the original one

# trainingFile=$PROJECT_ROOT/augmented_data/eda_train_subtask1.csv
trainingFile=$PROJECT_ROOT/CausalNewsCorpus/data/V2/train_subtask1.csv

python $PROJECT_ROOT/source/train_st1.py                                                                    \
--learning_rate $lr                                                                                         \
--train_file $trainingFile                                                                                  \
--do_train                                                                                                  \
--validation_file $PROJECT_ROOT/CausalNewsCorpus/data/V2/dev_subtask1.csv                                   \
--do_eval                                                                                                   \
--test_file $PROJECT_ROOT/CausalNewsCorpus/data/V2/test_subtask1_text.csv                                   \
--do_predict                                                                                                \
--num_train_epochs $numEpochs                                                                               \
--per_device_train_batch_size $batchSize                                                                    \
--per_device_eval_batch_size $batchSize                                                                     \
--seed $seed                                                                                                \
--output_dir $outputDir                                                                                     \
--model_name_or_path $modelNameOrPath                                                                       \
--use_weighted_ce $useWeightedCE
