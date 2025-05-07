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
from pydp.algorithms import numerical_mechanisms as dp_mechanisms
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
    """Creates PyDP partition selection object."""
    epsilon = pipeline_dp.dp_computations.gaussian_eps(sigma,
                                                       thresholding_delta)
    return create_partition_selection_strategy(
        pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING, epsilon,
        2 * thresholding_delta, max_partitions_contributed, pre_threshold)


def create_laplace_thresholding(
        sigma: float,
        thresholding_delta: float,
        max_partitions_contributed: int,
        pre_threshold: Optional[int] = None) -> "PartitionSelectionStrategy":
    """Creates PyDP partition selection object."""
    b = sigma / math.sqrt(2)
    epsilon = 1 / b
    return create_partition_selection_strategy(
        pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING, epsilon,
        thresholding_delta, max_partitions_contributed, pre_threshold)
