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
"""Utility Analysis cross partition combiners."""
import copy
import dataclasses

import pipeline_dp
from analysis import metrics
from typing import Optional
import math


def _sum_metrics_to_data_dropped(
        sum_metrics: metrics.SumMetrics,
        dp_metric: pipeline_dp.Metrics) -> Optional[metrics.DataDropInfo]:
    """Finds Data drop information from per-partition metrics."""
    # TODO(dvadym): implement
    return None


def _create_contribution_bounding_errors(
        sum_metrics: metrics.SumMetrics) -> metrics.ContributionBoundingErrors:
    """Creates ContributionBoundingErrors from per-partition metrics."""
    l0_mean = sum_metrics.expected_cross_partition_error
    l0_var = sum_metrics.std_cross_partition_error**2
    l0_mean_var = metrics.MeanVariance(mean=l0_mean, var=l0_var)
    linf_min = sum_metrics.per_partition_error_min
    linf_max = sum_metrics.per_partition_error_max
    return metrics.ContributionBoundingErrors(l0=l0_mean_var,
                                              linf_min=linf_min,
                                              linf_max=linf_max)


def _sum_metrics_to_value_error(sum_metrics: metrics.SumMetrics,
                                keep_prob: float) -> metrics.ValueErrors:
    """Creates ValueErrors from per-partition metrics."""
    value = sum_metrics.sum
    bounding_errors = _create_contribution_bounding_errors(sum_metrics)
    mean = bounding_errors.l0.mean + bounding_errors.linf_min + bounding_errors.linf_max
    variance = sum_metrics.std_cross_partition_error**2 + sum_metrics.std_noise**2

    rmse = math.sqrt(mean**2 + variance)
    l1 = 0  # TODO(dvadym) compute it.
    rmse_with_dropped_partitions = keep_prob * rmse + (1 -
                                                       keep_prob) * abs(value)
    l1_with_dropped_partitions = 0  # TODO(dvadym) compute it.
    result = metrics.ValueErrors(
        bounding_errors=bounding_errors,
        mean=mean,
        variance=variance,
        rmse=rmse,
        l1=l1,
        rmse_with_dropped_partitions=rmse_with_dropped_partitions,
        l1_with_dropped_partitions=l1_with_dropped_partitions)
    return result


def _sum_metrics_to_metric_utility(
        sum_metrics: metrics.SumMetrics, dp_metric: pipeline_dp.Metrics,
        partition_keep_probability: float) -> metrics.MetricUtility:
    """Creates cross partition MetricUtility from 1 partition utility.

    Attributes:
        sum_metrics: per-partition utility metric.
        dp_metric: metric for which utility is computed (e.g. COUNT)
        partition_keep_probability: partition selection probability.
    """
    assert dp_metric != pipeline_dp.Metrics.SUM, "Cross-partition metrics are not implemented for Sum"
    is_empty_public = sum_metrics.sum == 0  # it wouldn't work for SUM.
    data_dropped = _sum_metrics_to_data_dropped(sum_metrics, dp_metric)
    absolute_error = _sum_metrics_to_value_error(
        sum_metrics, keep_prob=partition_keep_probability)
    if sum_metrics.sum == 0:
        # When the actual value is 0, the relative error can't be computed. Set
        # relative errors to 0 in order not to influence aggregated relative
        # error.
        relative_error = metrics.ValueErrors.get_empty()
    else:
        relative_error = absolute_error.to_relative(sum_metrics.sum)
    return metrics.MetricUtility(
        metric=dp_metric,
        num_dataset_partitions=1,
        num_non_public_partitions=0,  # todo(dvadym): to implement it.
        num_empty_partitions=1 if is_empty_public else 0,
        noise_std=sum_metrics.std_noise,
        noise_kind=sum_metrics.noise_kind,
        ratio_data_dropped=data_dropped,
        absolute_error=absolute_error,
        relative_error=relative_error)
