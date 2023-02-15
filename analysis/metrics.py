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
class SumMetrics:
    """Stores per-partition metrics for SUM utility analysis.

    It is also used to store COUNT and PRIVACY_ID_COUNT per-partition metrics.

    Attributes:
        sum: actual sum of contributions per partition.
        per_partition_error_min: the amount of error due to contribution min clipping.
        per_partition_error_max: the amount of error due to contribution max clipping.
        expected_cross_partition_error: the expected amount of error due to cross-partition contribution bounding.
        std_cross_partition_error: the standard deviation of the error due to cross-partition contribution bounding.
        std_noise: the noise standard deviation.
        noise_kind: the type of noise used.

    s.t. the following holds (where E stands for Expectation):
    E(sum_after_contribution_bounding) = sum + E(error)
    where E(error) = per_partition_error_min + per_partition_error_max + expected_cross_partition_error
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
    SUM = 'sum'


@dataclass
class MeanVariance:
    mean: float
    var: float = 0


@dataclass
class ContributionBoundingErrors:
    l0: MeanVariance
    linf_mean: float
    linf_min_mean: float
    linf_max_mean: float


@dataclass
class ErrorsStatistics:  # ?
    # Contribution bounding
    bounding_errors: ContributionBoundingErrors

    # Errors
    rmse_error: float
    l1_error: float

    expected: float
    variance: float
    quantiles: List[float]  # ?


@dataclass
class AggregateErrorMetrics:  # ? UtilityAnalysisRes
    """Stores aggregate cross-partition metrics for utility analysis.

    All attributes in this dataclass are averages across partitions; except for
    ratio_* attributes, which are simply the ratios of total data dropped
    aggregated across partitions.
    """
    metric_type: AggregateMetricType

    noise_std: float
    noise_kind: pipeline_dp.NoiseKind

    ratio_data_dropped_l0: float
    ratio_data_dropped_linf: float
    # This cannot be computed at PartitionSelectionMetrics and needs to be
    # computed for each aggregation separately, since it takes into account data
    # drop from contribution bounding and that is aggregation-specific.
    ratio_data_dropped_partition_selection: float

    absolute_error: ErrorsStatistics
    relative_error: ErrorsStatistics

    # The following error metrics include error from dropped partitions.
    #
    # Consider the following example with a single partition to see how they are
    # different from abs/rel_error_expected metrics:
    #
    # Given 1 partition with probability_to_keep=0.4, actual_count=100,
    # abs_error_expected=-50;
    # -> abs_error_expected = (0.4*-50)/0.4=-50
    # -> abs_error_expected_w_dropped_partitions = 0.4*-50+0.6*-100=-80
    #
    # When public partitions are used, these will be exactly equal to
    # abs/rel_error_expected.
    error_expected_w_dropped_partitions: float
    rel_error_expected_w_dropped_partitions: float

    # RMSE = sqrt(bias**2 + variance), more details in
    # https://en.wikipedia.org/wiki/Bias-variance_tradeoff.
    def absolute_rmse(self) -> float:
        return math.sqrt(self.error_expected**2 + self.error_variance)

    def relative_rmse(self) -> float:
        return math.sqrt(self.rel_error_expected**2 + self.rel_error_variance)


@dataclass
class PartitionSelectionMetrics:
    """Stores aggregate metrics about partition selection."""

    num_partitions: float
    dropped_expected: float
    dropped_variance: float  # ? is it important


@dataclass
class AggregateMetrics:
    """Stores result of the utility analysis for specific input parameters.

    Attributes:
        input_aggregate_params: input parameters for which this utility analysis
          was computed.

        count_metrics: utility analysis of count. It is non None, if Count
          metric is in input_aggregate_params.metrics.
        privacy_id_count_metrics: utility analysis of sum. It is non None, if
          Sum  metric is in input_aggregate_params.metrics.
        partition_selection_metrics: utility analysis of selected partition. It
          is not None if the utility analysis is for private partition selection.
    """
    input_aggregate_params: pipeline_dp.AggregateParams

    count_metrics: Optional[AggregateErrorMetrics] = None
    privacy_id_count_metrics: Optional[AggregateErrorMetrics] = None
    partition_selection_metrics: Optional[PartitionSelectionMetrics] = None
