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

import pipeline_dp
from analysis import metrics
import dataclasses
from typing import List, Optional, Tuple
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
    """Creates cross-partition MetricUtility from 1 partition utility.

    Attributes:
        sum_metrics: per-partition utility metric.
        dp_metric: metric for which utility is computed (e.g. COUNT)
        partition_keep_probability: partition selection probability.
    """
    assert dp_metric != pipeline_dp.Metrics.SUM, "Cross-partition metrics are not implemented for SUM"
    # The next line does not work for SUM because the user can contribute 0.
    is_empty_public = sum_metrics.sum == 0
    data_dropped = _sum_metrics_to_data_dropped(sum_metrics, dp_metric)
    absolute_error = _sum_metrics_to_value_error(
        sum_metrics, keep_prob=partition_keep_probability)
    relative_error = absolute_error.to_relative(sum_metrics.sum)

    return metrics.UtilityReport(metric=dp_metric,
                                 noise_std=sum_metrics.std_noise,
                                 noise_kind=sum_metrics.noise_kind,
                                 ratio_data_dropped=data_dropped,
                                 absolute_error=absolute_error,
                                 relative_error=relative_error)


def _create_partition_metrics_for_public_partitions(
        is_empty_partition: bool) -> metrics.PartitionMetrics:
    result = metrics.PartitionMetrics(public_partitions=True,
                                      num_dataset_partitions=0,
                                      num_non_public_partitions=0,
                                      num_empty_partitions=0)
    if is_empty_partition:
        result.num_empty_partitions = 1
    else:
        result.num_dataset_partitions = 1
    return result


def _create_partition_metrics_for_private_partitions(
        prob_keep: float) -> metrics.PartitionMetrics:
    kept_partitions = metrics.MeanVariance(mean=prob_keep,
                                           var=prob_keep * (1 - prob_keep))
    return metrics.PartitionMetrics(public_partitions=False,
                                    num_partitions=1,
                                    kept_partitions=kept_partitions)


def _add_dataclasses_by_fields(dataclass1, dataclass2,
                               fields_to_ignore: List[str]) -> None:
    """Recursively adds all numerical fields of one dataclass to another.

    The result is stored in dataclass1. dataclass2 is unmodified.

    Assumptions:
      1. dataclass1 and dataclass2 are instances of the same dataclass type.
      2. all fields which should be processed (i.e. not in fields_to_ignore)
        are either dataclasses or support + operator.
      3. For all dataclasses fields assumptions 1,2 apply.

    Attributes:
        dataclass1, dataclass2: instances of the same dataclass type.
        fields_to_ignore: field names which should be ignored.
    """
    assert type(dataclass1) == type(
        dataclass2
    ), f"type(dataclass1) = {type(dataclass1)} != type(dataclass2) = {type(dataclass2)} must have the same types, their types {type(dataclass1)} and {type(dataclass2)}"
    fields = dataclasses.fields(dataclass1)
    for field in fields:
        if field.name in fields_to_ignore:
            continue
        value1 = getattr(dataclass1, field.name)
        if value1 is None:
            continue
        value2 = getattr(dataclass2, field.name)
        if dataclasses.is_dataclass(value1):
            _add_dataclasses_by_fields(value1, value2, fields_to_ignore)
            continue
        setattr(dataclass1, field.name, value1 + value2)


def _multiply_float_dataclasses_field(dataclass, factor: float) -> None:
    """Recursively multiplies all float fields of the dataclass by given number.

    Warning: it modifies 'dataclass' argument.
    """
    fields = dataclasses.fields(dataclass)
    for field in fields:
        value = getattr(dataclass, field.name)
        if value is None:
            continue
        if field.type is float:
            setattr(dataclass, field.name, value * factor)
        if dataclasses.is_dataclass(value):
            _multiply_float_dataclasses_field(value, factor)


def _per_partition_to_cross_partition_metrics(
        per_partition_utility: metrics.PerPartitionMetrics,
        dp_metrics: List[pipeline_dp.Metrics],
        public_partitions: bool) -> metrics.UtilityReport:
    """Converts per-partition to cross-partition utility metrics."""
    # Fill partition selection metrics.
    if public_partitions:
        prob_to_keep = 1
        partition_metrics = _create_partition_metrics_for_public_partitions()
    else:
        prob_to_keep = per_partition_utility.probability_to_keep
        partition_metrics = _create_partition_metrics_for_private_partitions(
            prob_to_keep)
    # Fill metric errors.
    metric_errors = None
    if dp_metrics:
        assert len(per_partition_utility.metric_errors) == len(dp_metrics)
        metric_errors = []
        for metric_error, dp_metric in zip(per_partition_utility.metric_errors,
                                           dp_metrics):
            metric_errors.append(
                _sum_metrics_to_metric_utility(metric_error, dp_metric,
                                               prob_to_keep))

    return metrics.UtilityReport(input_aggregate_params=None,
                                 partition_metrics=partition_metrics,
                                 metric_errors=metric_errors)


def _merge_partition_metrics(metrics1: metrics.PartitionMetrics,
                             metrics2: metrics.PartitionMetrics) -> None:
    """Merges cross-partition utility metrics.

    Warning: it modifies 'metrics1' argument.
    """
    _add_dataclasses_by_fields(metrics1, metrics2,
                               ["public_partitions", "strategy"])


def _merge_metric_utility(utility1: metrics.MetricUtility,
                          utility2: metrics.MetricUtility) -> None:
    """Merges cross-partition metric utilities.

    Warning: it modifies 'utility1' argument.
    """
    _add_dataclasses_by_fields(utility1, utility2,
                               ["metric", "noise_std", "noise_kind"])


def _merge_utility_reports(report1: metrics.UtilityReport,
                           report2: metrics.UtilityReport) -> None:
    """Merges cross-partition utility reports.

    Warning: it modifies 'report1' argument.
    """
    _merge_partition_metrics(report1.partition_selection,
                             report2.partition_selection)
    if report1.metric_errors is None:
        return
    assert len(report1.metric_errors) == len(report2.metric_errors)
    for utility1, utility2 in zip(report1.metric_errors, report2.metric_errors):
        _merge_metric_utility(utility1, utility2)


def _normalize_utility_report(report: metrics.UtilityReport,
                              denominator: float):
    _multiply_float_dataclasses_field(report, 1.0 / denominator)


class CrossPartitionCompoundCombiner(pipeline_dp.combiners.Combiner):
    """A compound combiner for aggregating error metrics across partitions"""

    def __init__(self, dp_metrics: List[pipeline_dp.Metrics],
                 public_partitions: bool):
        self._dp_metrics = dp_metrics
        self._public_partitions = public_partitions

    def create_accumulator(
            self,
            metrics: metrics.PerPartitionMetrics) -> metrics.UtilityReport:
        return _per_partition_to_cross_partition_metrics(
            metrics, self._dp_metrics, self._public_partitions)

    def merge_accumulators(
            self, report1: metrics.UtilityReport,
            report2: metrics.UtilityReport) -> metrics.UtilityReport:
        """Merges UtilityReports."""
        _merge_utility_reports(report1, report2)
        return report1

    def compute_metrics(self,
                        report: metrics.UtilityReport) -> metrics.UtilityReport:
        """Normalizes and returns UtilityReport."""
        expected_num_output_partitions = 1  # todo
        _normalize_utility_report(report, expected_num_output_partitions)
        return report

    def metrics_names(self):
        return []  # Not used for utility analysis

    def explain_computation(self):
        return None  # Not used for utility analysis
