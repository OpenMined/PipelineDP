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


def create_partition_selection_strategy(
        strategy: pipeline_dp.PartitionSelectionStrategy, epsilon: float,
        delta: float,
        max_partitions_contributed: int) -> "PartitionSelectionStrategy":

    if strategy == pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC:
        create_strategy = partition_selection.create_truncated_geometric_partition_strategy
    elif strategy == pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING:
        create_strategy = partition_selection.create_laplace_partition_strategy
    elif strategy == pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING:
        create_strategy = partition_selection.create_gaussian_partition_strategy
    else:
        raise ValueError(f"Unknown partition selection strategy {strategy}")

    return create_strategy(epsilon, delta, max_partitions_contributed)
