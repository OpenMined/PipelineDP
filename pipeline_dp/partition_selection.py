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

import pipeline_dp
import pydp.algorithms.partition_selection as partition_selection
from typing import Optional
import math

PARTITION_STRATEGY_ENUM_TO_STR = {
    pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC:
        "truncated_geometric",
    pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING:
        "laplace",
    pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING:
        "gaussian"
}


def create_partition_selection_strategy(
        strategy: pipeline_dp.PartitionSelectionStrategy,
        epsilon: float,
        delta: float,
        max_partitions_contributed: int,
        pre_threshold: Optional[int] = None) -> "PartitionSelectionStrategy":
    """Creates PyDP partition selection object."""
    strategy_name = PARTITION_STRATEGY_ENUM_TO_STR[strategy]

    if pre_threshold is None:
        return partition_selection.create_partition_strategy(
            strategy_name, epsilon, delta, max_partitions_contributed)
    else:
        return partition_selection.create_partition_strategy(
            strategy_name, epsilon, delta, max_partitions_contributed,
            pre_threshold)


def create_gaussian_thresholding(
        sigma: float,
        thresholding_delta: float,
        max_partitions_contributed: int,
        pre_threshold: Optional[int] = None) -> "PartitionSelectionStrategy":
    """Creates PyDP Gaussian partition selection object.

    Gaussian thresholding with parameters sigma and thresholding_delta is the
    partition selection mechanism:
      if num_privacy_units + N(0, sigma^2) >= threshold:
        return "release"
      else:
        return "not-release"
    Where `threshold` is chosen, such that the probability of releasing of a
    partition with 1 privacy unit is less or equal to thresholding_delta, i.e.
      P(1 + N(0, sigma^2) >= threshold) <= thresholding_delta.

    Args:
       sigma: the standard deviation of Gaussian noise
       thresholding_delta: upper bound on the probability of releasing of a
         partition with 1 privacy unit
       max_partitions_contributed: A bound on the number of partitions to
          which one unit of privacy (e.g., a user) can contribute.
       pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection.
    """
    # Now PyDP supports only creating partition selection by epsilon, delta.
    # So we need to find (epsilon, delta), such that the created object has
    # the corresponding noise scale:
    # delta = 2 * threshold_delta (because Gaussian thresholding divides evenly
    # delta for the noise and threshold)
    # From sigma and noise_delta we can find epsilon of corresponding Gaussian
    # mechanism.
    epsilon = pipeline_dp.dp_computations.gaussian_epsilon(
        sigma, thresholding_delta)
    total_delta = 2 * thresholding_delta
    return create_partition_selection_strategy(
        pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING, epsilon,
        total_delta, max_partitions_contributed, pre_threshold)


def create_laplace_thresholding(
        sigma: float,
        thresholding_delta: float,
        max_partitions_contributed: int,
        pre_threshold: Optional[int] = None) -> "PartitionSelectionStrategy":
    """Creates PyDP Laplace partition selection object.

    Laplace thresholding with parameters b and thresholding_delta is the
     partition selection mechanism:
       if num_privacy_units + Lap(b) >= threshold:
         return "release"
       else:
         return "not-release"
     Where `threshold` is chosen, such that the probability of releasing of a
     partition with 1 privacy unit is less equal to thresholding_delta, i.e.
       P(1 + Lap(b) >= threshold) <= thresholding_delta.

     Args:
        sigma: the standard deviation of Laplace noise
        thresholding_delta: upper bound on the probability of releasing of a
          partition with 1 privacy unit
        max_partitions_contributed: A bound on the number of partitions to
           which one unit of privacy (e.g., a user) can contribute.
        pre_threshold: the minimum amount of privacy units which are required
          for keeping a partition in private partition selection.
     """
    # Now PyDP supports only creating partition selection by epsilon, delta.
    # So we need to find (epsilon, delta), such that the created object has
    # the corresponding noise scale:
    # delta = threshold_delta
    # epsilon = 1 / laplace_b
    # laplace_b = sigma/sqrt(2)
    b = sigma / math.sqrt(2)
    epsilon = 1 / b
    return create_partition_selection_strategy(
        pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING, epsilon,
        thresholding_delta, max_partitions_contributed, pre_threshold)
