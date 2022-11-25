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
from enum import Enum

import pipeline_dp
from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class CountMetrics:
    """Stores metrics for the count utility analysis.

  Attributes:
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

  Attributes:
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


class AggregateMetricType(Enum):
    PRIVACY_ID_COUNT = 'privacy_id_count'
    COUNT = 'count'


@dataclass
class AggregateErrorMetrics:
    """Stores aggregate metrics for utility analysis.

  All attributes in this dataclass are averages across partitions; except for
  ratio_* attributes, which are simply the ratios of total data dropped
  aggregated across partitions.
  """
    metric_type: AggregateMetricType

    ratio_data_dropped_l0: float
    ratio_data_dropped_linf: float
    # This cannot be computed at PartitionSelectionMetrics and needs to be
    # computed for each aggregation separately, since it takes into account data
    # drop from contribution bounding and that is aggregation-specific.
    ratio_data_dropped_partition_selection: float

    abs_error_l0_expected: float
    abs_error_linf_expected: float
    abs_error_expected: float
    abs_error_l0_variance: float
    abs_error_variance: float
    abs_error_quantiles: List[float]
    rel_error_l0_expected: float
    rel_error_linf_expected: float
    rel_error_expected: float
    rel_error_l0_variance: float
    rel_error_variance: float
    rel_error_quantiles: List[float]

    # These metrics take into account loss due to dropping partitions when
    # computing expectation.
    #
    # When public partitions are used, these will be exactly equal to
    # abs/rel_error_expected.
    #
    # When private partitions are used, see the following example with a single
    # partition on how they are different:
    #
    # Given 1 partition with probability_to_keep=0.4, actual_count=100,
    # abs_error_expected=-50;
    # -> abs_error_expected = (0.4*-50)/0.4=-50
    # -> abs_error_expected_w_dropped_partitions = 0.4*-50+0.6*-100=-80
    abs_error_expected_w_dropped_partitions: float
    rel_error_expected_w_dropped_partitions: float

    noise_variance: float

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


@dataclass
class AggregateMetrics:
    """Stores aggregate metrics for utility analysis."""
    count_metrics: Optional[AggregateErrorMetrics] = None
    privacy_id_count_metrics: Optional[AggregateErrorMetrics] = None
    partition_selection_metrics: Optional[PartitionSelectionMetrics] = None
