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
from pipeline_dp import combiners
from analysis import data_structures
from analysis import metrics
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


def _partition_selection_per_to_cross_partition(
        prob_keep: float) -> metrics.PrivatePartitionSelectionUtility:
    return metrics.PrivatePartitionSelectionUtility(
        strategy=None,
        num_partitions=1,
        dropped_partitions=metrics(mean=prob_keep,
                                   var=prob_keep * (1 - prob_keep)),
        ratio_dropped_data=0)  # todo: implement


def _per_partition_to_cross_partition_utility(
        per_partition_utility: metrics.PerPartitionUtility,
        dp_metrics: List[pipeline_dp.Metrics],
        public_partition: bool) -> metrics.UtilityReport:
    # Partition selection
    prob_to_keep = per_partition_utility.partition_selection_probability_to_keep
    partition_selection_utility = None
    if not public_partition:
        partition_selection_utility = _partition_selection_per_to_cross_partition(
            prob_to_keep)
    # Metric errors
    metric_errors = None
    if dp_metrics:
        metric_errors = []
        assert len(per_partition_utility.metric_errors) == len(dp_metrics)
        for metric_error, dp_metric in zip(per_partition_utility.metric_errors,
                                           dp_metrics):
            metric_errors.append(
                _sum_metrics_to_metric_utility(metric_error, dp_metric,
                                               prob_to_keep))

    return metrics.UtilityReport(
        per_partition_utility=partition_selection_utility,
        metric_errors=metric_errors)


def _add_dataclasses_by_fields(lhs, rhs, fields_to_ignore):
    assert type(lhs) == type(rhs)
    fields = dataclasses.fields(lhs)
    for field in fields:
        if field.name in fields_to_ignore:
            continue
        value1 = getattr(lhs, field.name)
        value2 = getattr(rhs, field.name)
        if dataclasses.is_dataclass(value1):
            _add_dataclasses_by_fields(value1, value2, fields_to_ignore)
            continue
        setattr(lhs, field.name, value1 + value2)
    return lhs  # needed?


def _multiply_float_dataclasses_field(dataclass, factor: float):
    fields = dataclasses.fields(dataclass)
    for field in fields:
        value = getattr(dataclass, field.name)
        if value is None:
            continue
        if value is float:
            setattr(dataclass, field.name, value * factor)
        if dataclasses.is_dataclass(value):
            _multiply_float_dataclasses_field(value, factor)
            continue


def _merge_partition_selection_utilities(
        utility1: metrics.PrivatePartitionSelectionUtility,
        utility2: metrics.PrivatePartitionSelectionUtility) -> None:
    utility1.num_partitions += utility2.num_partitions
    utility1.dropped_partitions.mean += utility2.dropped_partitions.mean
    utility1.dropped_partitions.var += utility2.dropped_partitions.var
    # todo(dvadym): implement for ratio_dropped_data


def _merge_metric_utility(utility1: metrics.MetricUtility,
                          utility2: metrics.MetricUtility) -> None:
    _add_dataclasses_by_fields(utility1, utility2,
                               ["metric", "noise_std", "noise_kind"])


def _merge_utility_reports(report1: metrics.UtilityReport,
                           report2: metrics.UtilityReport):
    _merge_partition_selection_utilities(report1.partition_selection,
                                         report2.partition_selection)
    if report1.metric_errors is None:
        return
    assert len(report1.metric_errors) == len(report2.metric_errors)
    for utility1, utility2 in zip(report1.metric_errors, report2.metric_errors):
        _merge_metric_utility(utility1, utility2)


def _normalize_utility_report(report: metrics.UtilityReport,
                              denominator: float):
    _multiply_float_dataclasses_field(report, 1.0 / denominator)


class CrossPartitionCompoundCombiner(combiners.Combiner):
    """A compound combiner for aggregating error metrics across partitions"""
    AccumulatorType = Tuple[metrics.UtilityReport]
    OutputType = AccumulatorType

    def __init__(self, options: data_structures.UtilityAnalysisOptions):
        self._options = options

    def create_accumulator(
        self, per_partition_utilities: Tuple[metrics.PerPartitionUtility]
    ) -> AccumulatorType:
        return tuple(
            _per_partition_to_cross_partition_utility(utility)
            for utility in per_partition_utilities)

    def merge_accumulators(self, accumulator1: AccumulatorType,
                           accumulator2: AccumulatorType) -> AccumulatorType:
        """Merges the accumulators and returns accumulator."""
        for report1, report2 in zip(accumulator1, accumulator2):
            _merge_utility_reports(report1, report2)
        return accumulator1

    def compute_metrics(self, utility_reports: AccumulatorType) -> OutputType:
        """Computes and returns the result of aggregation."""
        multi_aggregate_params = list(
            data_structures.get_aggregate_params(self._options))
        result = []
        for report, params in zip(utility_reports, multi_aggregate_params):
            _normalize_utility_report(report, 1.0)  # todo 1.0 -> to_proper
            report.input_aggregate_params = params
            result.append(report)
        return result

    def metrics_names(self):
        return []  # Not used

    def explain_computation(self):
        return None  # Not used
