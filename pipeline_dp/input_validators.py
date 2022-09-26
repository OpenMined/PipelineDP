# Copyright 2022 OpenMined.
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
"""Helper functions for validating input arguments."""


def validate_epsilon_delta(epsilon: float, delta: float, obj_name: str):
    """Helper function to validate the epsilon and delta parameters.

  Args:
      epsilon: The epsilon value to validate.
      delta: The delta value to validate.

  Raises:
      A ValueError if either epsilon or delta are out of range.
  """
    if epsilon <= 0:
        raise ValueError(
            f"{obj_name}: epsilon must be positive, not {epsilon}.")
    if delta < 0:
        raise ValueError(
            f"{obj_name}: delta must be non-negative, not {delta}.")
    if delta >= 1:
        raise ValueError(f"{obj_name}: delta must be less than 1, not {delta}.")
