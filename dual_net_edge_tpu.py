# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This file contains an implementation of dual_net.py for Google's EdgeTPU.
It can only be used for inference and requires a specially quantized and
compiled model file.

For more information see https://coral.withgoogle.com
"""

from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from pycoral.adapters import common

import features as features_lib
import go

# pylint: disable=missing-function-docstring


def extract_agz_features(position):
    return features_lib.extract_features(position, features_lib.AGZ_FEATURES)


class DualNetworkEdgeTpu():
    """DualNetwork implementation for Google's EdgeTPU."""

    def __init__(self, save_file):

        # self.engine = BasicEngine(save_file)

        self.interpreter = make_interpreter(save_file)
        self.interpreter.allocate_tensors()

        self.board_size = go.N
        self.output_policy_size = self.board_size**2 + 1

    def run_many(self, positions):
        """Runs inference on a list of position."""
        processed = map(extract_agz_features, positions)
        probabilities = []
        values = []
        for state in processed:
            assert state.shape == (self.board_size, self.board_size, 17)

            run_inference(self.interpreter, state.flatten())

            policy_scale = self.interpreter.get_output_details()[
                0]['quantization'][0]
            value_scale = self.interpreter.get_output_details()[
                1]['quantization'][0]

            policy_zero = self.interpreter.get_output_details()[
                0]['quantization'][1]
            value_zero = self.interpreter.get_output_details()[
                1]['quantization'][1]

            policy_output = (common.output_tensor(self.interpreter, 0)[
                             0] - policy_zero) * policy_scale
            value_output = (common.output_tensor(self.interpreter, 1)[
                            0] - value_zero) * value_scale

            probabilities.append(policy_output)
            values.append(value_output)

        return probabilities, values
