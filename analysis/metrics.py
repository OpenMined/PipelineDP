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
        aggregation: DP aggregation for which this analysis was performed. It
          can be COUNT, PRIVACY_ID_COUNT or SUM.
        sum: Non-DP sum of contributions to the aggregated metric being
          analyzed. In case of SUM, this field stores the sum of all values
          contributed by privacy IDs, in  case of COUNT, it is the count of
          contributed values, and in case of PRIVACY_ID_COUNT, the field
          contains count of privacy IDs that contribute to a partition.
        clipping_to_min_error: the amount of error due to contribution min
          clipping.
        clipping_to_max_error: the amount of error due to contribution max
          clipping.
        expected_l0_bounding_error: the mathematical expectation of error due to
          cross-partition contribution bounding.
        std_l0_bounding_error : the standard deviation of the error due to
          cross-partition contribution bounding.
        std_noise: the noise standard deviation.
        noise_kind: the type of noise used.

    s.t. the following holds (where E stands for Expectation):
    E(sum_after_contribution_bounding) = sum + E(error)
    where E(error) = clipping_to_min_error + clipping_to_max_error + expected_l0_bounding_error
    """
    aggregation: pipeline_dp.Metric
    sum: float
    clipping_to_min_error: float
    clipping_to_max_error: float
    expected_l0_bounding_error: float
    std_l0_bounding_error: float
    std_noise: float
    noise_kind: pipeline_dp.NoiseKind


@dataclass
class Statistics:
    privacy_id_count: int
    count: int


@dataclass
class PerPartitionMetrics:
    partition_selection_probability_to_keep: float
    statistics: Statistics
    metric_errors: Optional[List[SumMetrics]] = None


@dataclass
class MeanVariance:
    mean: float
    var: float


@dataclass
class ContributionBoundingErrors:
    """Contains an error breakdown by types of contribution bounding.

    Attributes:
        l0: max_partition_contributed (aka l0) bounding error. The output of l0
          bounding is a random variable for each partition. Its distribution is
          close to normal when number of contribution per partition is large.
        linf_min & linf_max: represents error due to min & max contribution
          bounding, respectively (only populated for Sum metrics). It is
          deterministic for each partition.
    """
    l0: MeanVariance
    linf_min: float
    linf_max: float

    def to_relative(self, value: float) -> 'ContributionBoundingErrors':
        """Converts from the absolute to the relative error dividing by actual value."""
        l0_rel_mean_var = MeanVariance(self.l0.mean / value,
                                       self.l0.var / value**2)
        return ContributionBoundingErrors(l0=l0_rel_mean_var,
                                          linf_min=self.linf_min / value,
                                          linf_max=self.linf_max / value)


@dataclass
class ValueErrors:
    """Errors between actual and dp metric.

    This class describes breakdown of errors for (dp_value - actual_value),
    where value can be a metric like count, sum etc. The value error is a random
    variable and it comes from different sources - contribution bounding error
    and DP noise. This class contains different error metrics.

    All attributes correspond to the errors computed per partition and then
    averaged across partitions, e.g.
      rmse_per_partition = sqrt(E(dp_value - actual_value)^2)
      self.rmse = mean(rmse_per_partition)

    Attributes:
        bounding_errors: contribution bounding errors.
        mean: averaged across partitions E(dp_value - actual_value), aka
         statistical bias.
        variance: averaged across partitions Var(dp_value - actual_value).
        rmse: averaged across partitions sqrt(E(dp_value - actual_value)^2).
        l1: averaged across partitions E|dp_value - actual_value|.
        with_dropped_partitions: error which takes into consideration partitions
          dropped due to partition selection. See example below.
    """
    bounding_errors: ContributionBoundingErrors
    mean: float
    variance: float

    rmse: float
    l1: float

    # The following error metrics include error from dropped partitions for
    # private partition selection. For example:
    #    rmse = sqrt(E(actual_value-dp_output)^2) = f(actual_value-dp_output).
    # For the partition with probability to keep = p.
    #    rmse_with_dropped_partitions = p*rmse + (1-p)*f(actual_value-0).
    rmse_with_dropped_partitions: float
    l1_with_dropped_partitions: float

    def to_relative(self, value: float):
        """Converts from absolute to relative error dividing by actual_value."""
        if value == 0:
            # When the actual value is 0, the relative error can't be computed.
            # Set relative errors to 0 in order not to influence aggregated
            # relative error.
            empty_bounding = ContributionBoundingErrors(l0=MeanVariance(0, 0),
                                                        linf_min=0,
                                                        linf_max=0)
            return ValueErrors(bounding_errors=empty_bounding,
                               mean=0,
                               variance=0,
                               rmse=0,
                               l1=0,
                               rmse_with_dropped_partitions=0,
                               l1_with_dropped_partitions=0)
        return ValueErrors(
            self.bounding_errors.to_relative(value),
            mean=self.mean / value,
            variance=self.variance / value**2,
            rmse=self.rmse / value,
            l1=self.l1 / value,
            rmse_with_dropped_partitions=self.rmse_with_dropped_partitions /
            value,
            l1_with_dropped_partitions=self.l1_with_dropped_partitions / value)


@dataclass
class DataDropInfo:
    """Information about the data dropped during different DP stages.

    Attributes:
        l0: ratio of data dropped during of l0 contribution bounding.
        linf: ratio of data dropped during of linf contribution bounding.
        partition_selection: ratio of data dropped because of partition
          selection.
    """
    l0: float
    linf: float

    # This cannot be computed at PartitionSelectionUtility and needs to be
    # computed for each aggregation separately, since it takes into account data
    # drop from contribution bounding and that is aggregation-specific.
    partition_selection: float


@dataclass
class MetricUtility:
    """Stores aggregate cross-partition metrics for utility analysis.

    It contains utility analysis for 1 DP metric (COUNT, SUM etc).

    Attributes:
        metric: DP metric for which this analysis was performed.
        noise_std: the standard deviation of added noise.
        noise_kind: the noise kind (Laplace or Gaussian)
        ratio_data_dropped: the information about dropped data.
        absolute_error: error in terms of (dp_value - actual_value).
        relative_error: error in terms of (dp_value - actual_value)/actual_value.
    """
    metric: pipeline_dp.Metrics

    # Noise information.
    noise_std: float
    noise_kind: pipeline_dp.NoiseKind

    # Dropped data breakdown.
    ratio_data_dropped: Optional[DataDropInfo]

    # Value errors
    absolute_error: ValueErrors
    relative_error: ValueErrors


@dataclass
class PartitionsInfo:
    """Stores aggregate metrics about partitions and partition selection.

    Attributes:
        public_partitions: true if public partitinos are used.
        num_dataset_partitions: the number of partitions in dataset.
        num_non_public_partitions: the number of partitions dropped because
          of public partitions.
        num_empty_partitions: the number of empty partitions added because of
          public partitions.
        strategy: Private partition selection strategy. None if public
          partitions are used.
        kept_partitions: Mean and Variance of the number of kept partitions.
    """
    public_partitions: bool

    # Common
    num_dataset_partitions: int

    # Public partitions
    num_non_public_partitions: Optional[int] = None
    num_empty_partitions: Optional[int] = None

    # Private partition selection
    strategy: Optional[pipeline_dp.PartitionSelectionStrategy] = None
    kept_partitions: Optional[MeanVariance] = None


@dataclass
class UtilityReport:
    """Stores result of the utility analysis for specific input parameters.

    Attributes:
        configuration_index: the index of the input parameter configuration for
          which this report was computed.
        partition_metrics: utility analysis of selected partition.
        metric_errors: utility analysis of metrics (e.g. COUNT, SUM,
          PRIVACY_ID_COUNT).
        utility_report_histogram:
    """
    configuration_index: int

    partitions_info: PartitionsInfo
    metric_errors: Optional[List[MetricUtility]] = None
    utility_report_histogram: Optional[List['UtilityReportBin']] = None


@dataclass
class UtilityReportBin:
    """Stores a bin for a histogram of UtilityReports by partition size.

    The partition size is the non-DP value of the metric whose utility analysis
    was computed. The metric can be COUNT, PRIVACY_ID_COUNT, SUM.

     Attributes:
        partition_size_from: lower bound of partitions size.
        partition_size_to: upper bound of partitions size.
        report: the result of utility analysis for partitions of size
          [partition_size_from, partition_size_to).
    """
    partition_size_from: int
    partition_size_to: int
    report: UtilityReport
