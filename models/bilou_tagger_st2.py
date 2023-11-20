# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the BERT- and CRF-based sequence tagging model used for Subtask 2.
"""

from typing import Iterator

import torch
from torch import nn
from transformers import AutoModel

from models.crf import CRF


class NER_CRF_Classifier(nn.Module):
    """
    A BERT-based sequence labeling model with a conditional random field as output layer.
    """

    def __init__(self, bert_model_name: str, labels: list[str]) -> None:
        """
        Initializes the model.

        Args:
            bert_model_name (str): Path (or name) of the BERT model.
            labels (list[str]): List of labels that is used to calculate the CRF dimensions.
        """
        super().__init__()
        self._bert_model: AutoModel = AutoModel.from_pretrained(bert_model_name)
        self._dropout: nn.Dropout = nn.Dropout(p=0.2)
        self._linear: nn.Linear = nn.Linear(
            in_features=self._bert_model.config.hidden_size, out_features=len(labels)
        )
        self._crf: CRF = CRF(num_tags=len(labels), batch_first=True)

    def get_lm_parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns the trainable parameters of the BERT-based language model.

        Yields:
            Iterator[torch.Tensor]: Trainable parameters
        """
        layers: list = [self._bert_model.parameters(), self._linear.parameters()]
        for lay in layers:
            for p in lay:
                yield p

    def get_crf_parameters(self) -> Iterator[torch.Tensor]:
        """
        Returns the trainable parameters of the CRF.

        Yields:
            Iterator[torch.Tensor]: Trainable parameters
        """
        for p in self._crf.parameters():
            yield p

    def _get_linear_output_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the unnormalized logits produced by the linear linear that is fed with BERT embeddings.

        Args:
            input_ids (torch.Tensor): BERT-based input IDs
            attention_mask (torch.Tensor): BERT-based attention mask
            token_type_ids (torch.Tensor): BERT-based token type IDs

        Returns:
            torch.Tensor: Unnormalized logits
        """
        embeddings: torch.Tensor = self._bert_model(
            input_ids, attention_mask, token_type_ids
        ).last_hidden_state
        embeddings = self._dropout(embeddings)
        logits: torch.Tensor = self._linear(embeddings)
        return logits

    def get_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        crf_mask: torch.Tensor,
        sorted_crf_mask: torch.Tensor,
        tag_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the loss between prediction and ground truth.

        Args:
            input_ids (torch.Tensor): BERT-based input IDs
            attention_mask (torch.Tensor): BERT-based attention mask
            token_type_ids (torch.Tensor): BERT-based token type IDs
            crf_mask (torch.Tensor): Mask that distinguishes "real" tokens and BERT-specific subword tokens
            sorted_crf_mask (torch.Tensor): CRF mask that is already sorted
            tag_sequence (torch.Tensor): Ground truth tag sequence

        Returns:
            torch.Tensor: Loss score
        """
        logits: torch.Tensor = self._get_linear_output_logits(
            input_ids, attention_mask, token_type_ids
        )
        pad_size: int = logits.shape[-2]
        seq_len: int = logits.shape[-1]
        sorted_logits: list[torch.Tensor] = [
            torch.stack([score for j, score in enumerate(l) if crf_mask[i][j]])
            for i, l in enumerate(logits)
        ]
        for i in range(len(sorted_logits)):
            for _ in range(len(sorted_logits[i]), pad_size):
                if len(sorted_logits[i]) < pad_size:
                    sorted_logits[i] = torch.cat(
                        [
                            sorted_logits[i],
                            torch.stack(
                                [
                                    torch.tensor([0] * seq_len).to(logits.device)
                                    for _ in range(len(sorted_logits[i]), pad_size)
                                ]
                            ),
                        ]
                    )
        return self._crf(
            torch.stack(sorted_logits), tag_sequence, sorted_crf_mask, reduction="mean"
        )

    def predict_tag_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        crf_mask: torch.Tensor,
        sorted_crf_mask: torch.Tensor,
    ) -> list[list[int]]:
        """
        Outputs the predicted tag sequence for the input tokens.

        Args:
            input_ids (torch.Tensor): BERT-based input IDs
            attention_mask (torch.Tensor): BERT-based attention mask
            token_type_ids (torch.Tensor): BERT-based token type IDs
            crf_mask (torch.Tensor): Mask that distinguishes "real" tokens and BERT-specific subword tokens
            sorted_crf_mask (torch.Tensor): CRF mask that is already sorted

        Returns:
            list[list[int]]: One list of label predictions for each input sequence
        """
        logits: torch.Tensor = self._get_linear_output_logits(
            input_ids, attention_mask, token_type_ids
        )
        pad_size: int = logits.shape[-2]
        seq_len: int = logits.shape[-1]
        sorted_logits: list[torch.Tensor] = [
            torch.stack([score for j, score in enumerate(l) if crf_mask[i][j]])
            if not all(crf_mask[i] == 0)
            else torch.tensor([]).to(input_ids.device)
            for i, l in enumerate(logits)
        ]
        for i in range(len(sorted_logits)):
            for _ in range(len(sorted_logits[i]), pad_size):
                if len(sorted_logits[i]) < pad_size:
                    sorted_logits[i] = torch.cat(
                        [
                            sorted_logits[i],
                            torch.stack(
                                [
                                    torch.tensor([0] * seq_len).to(logits.device)
                                    for _ in range(len(sorted_logits[i]), pad_size)
                                ]
                            ),
                        ]
                    )
        sorted_crf_mask[:, 0] = 1
        return self._crf.decode(torch.stack(sorted_logits), sorted_crf_mask)
