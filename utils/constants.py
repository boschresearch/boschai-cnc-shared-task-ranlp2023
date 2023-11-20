# Experiment resources related to the BoschAI participation at the CNC shared task (2023).
# Copyright (c) 2023 Robert Bosch GmbH
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains device and path constants.
"""

from torch.cuda import is_available

CPU: str = "cpu"
DEVICE: str = "cuda" if is_available() else CPU
