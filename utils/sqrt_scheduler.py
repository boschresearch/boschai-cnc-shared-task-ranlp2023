# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the learning rate scheduler.
"""
import math


class SqrtSchedule:
    """
    A noam learning rate scheduler that builds upon the scheduler used by Vaswani et al., (2017) in "Attention is all you need".
    """

    def __init__(self, ws: int) -> None:
        """
        Initializes the LR scheduler.

        Args:
            ws (int): Number of warmup steps
        """
        self._ws: int = ws
        self._d: int = math.sqrt(self._ws)
        self._inv_ws: float = 1 / (math.sqrt(self._ws) ** 3)

    def __call__(self, step: int) -> float:
        return 0 if step == 0 else (self._d * min(1 / math.sqrt(step), step * self._inv_ws))
