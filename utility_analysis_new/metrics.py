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
"""Dataclasses with Utility Analysis result metrics."""

import pipeline_dp
from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class CountMetrics:
    """Stores metrics for the count utility analysis.

  Args:
      count: actual count of contributions per partition.
      per_partition_error: the amount of error due to per-partition
      contribution bounding.
      expected_cross_partition_error: the expected amount of error due to
      cross-partition contribution bounding.
      std_cross_partition_error: the standard deviation of the error due to
      cross-partition contribution bounding.
      std_noise: the noise standard deviation.
      noise_kind: the type of noise used.
  """
    count: int
    per_partition_error: int
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class SumMetrics:
    """Stores metrics for the sum utility analysis.

  Args:
      sum: actual sum of contributions per partition.
      per_partition_error_min: the amount of error due to contribution min clipping.
      per_partition_error_max: the amount of error due to contribution max clipping.
      expected_cross_partition_error: the expected amount of error due to cross-partition contribution bounding.
      std_cross_partition_error: the standard deviation of the error due to cross-partition contribution bounding.
      std_noise: the noise standard deviation.
      noise_kind: the type of noise used.
  """
    sum: float
    per_partition_error_min: float
    per_partition_error_max: float
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class PrivacyIdCountMetrics:
    """Stores metrics for the privacy ID count utility analysis.

  Args:
      privacy_id_count: actual count of privacy id in a partition.
      expected_cross_partition_error: the estimated amount of error across partitions.
      std_cross_partition_error: the standard deviation of the contribution bounding error.
      std_noise: the noise standard deviation for DP count.
      noise_kind: the type of noise used.
  """
    privacy_id_count: int
    expected_cross_partition_error: float
    std_cross_partition_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class AggregateErrorMetrics:
    """Stores aggregate metrics for utility analysis.

  All attributes in this dataclass are averages across partitions.
  """

    abs_error_expected: float
    abs_error_variance: float
    abs_error_quantiles: List[float]
    rel_error_expected: float
    rel_error_variance: float
    rel_error_quantiles: List[float]

    # RMSE = sqrt(bias**2 + variance), more details in
    # https://en.wikipedia.org/wiki/Bias-variance_tradeoff.
    def absolute_rmse(self) -> float:
        return math.sqrt(self.abs_error_expected**2 + self.abs_error_variance)

    def relative_rmse(self) -> float:
        return math.sqrt(self.rel_error_expected**2 + self.rel_error_variance)


@dataclass
class PartitionSelectionMetrics:
    """Stores aggregate metrics about partition selection."""

    num_partitions: float
    dropped_partitions_expected: float
    dropped_partitions_variance: float
