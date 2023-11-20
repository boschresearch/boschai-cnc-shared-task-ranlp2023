# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the BERT-based binary classifier used for Subtask 1.
"""

import torch
from torch import nn
from transformers import AutoModel


class BinaryClassifier(nn.Module):
    """
    A BERT-based binary classifier with one or two output nodes.
    """

    def __init__(self, model_path: str, use_softmax: bool, *args, **kwargs) -> None:
        """
        Initializes the classifier.

        Args:
            model_path (str): Path (or name) of the BERT model.
            use_softmax (bool): If true, the model will output two softmax-normalized prediction scores, one for each class, instead of a single score.
        """
        super().__init__(*args, **kwargs)
        self._embedding_model: AutoModel = AutoModel.from_pretrained(model_path)
        self._binary_neuron: nn.Linear = nn.Linear(
            in_features=self._embedding_model.config.hidden_size,
            out_features=2 if use_softmax else 1,
        )
        self._sigmoid: nn.Sigmoid = nn.Sigmoid()
        self._softmax: nn.Softmax = nn.Softmax()
        self._use_softmax: bool = use_softmax

    def forward(self, input_tokens: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
        """
        A forward pass in the model.

        Args:
            input_tokens (torch.Tensor): BERT-based input tokens.
            attention_masks (torch.Tensor): BERT-based input mask.

        Returns:
            torch.Tensor: 1 sigmoid-normalized score or 2 softmax-normalized scores
        """
        embeddings = self._embedding_model(input_tokens, attention_masks).last_hidden_state[
            :, 0, :
        ]
        predictions = self._binary_neuron(embeddings)
        if self._use_softmax:
            return self._softmax(predictions)
        else:
            return self._sigmoid(predictions)
