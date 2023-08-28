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

import pipeline_dp
from analysis import metrics
import dataclasses
from typing import List, Tuple, Callable
import math


def _sum_metrics_to_data_dropped(
        sum_metrics: metrics.SumMetrics, partition_keep_probability: float,
        dp_metric: pipeline_dp.Metric) -> metrics.DataDropInfo:
    """Finds Data drop information from per-partition metrics."""

    # This function attributed the data that is dropped, to different reasons
    # how they are dropped.

    # 1. linf/l0 contribution bounding
    # Contribution bounding errors are negative, negate to keep data dropped
    # to be positive.
    linf_dropped = sum_metrics.clipping_to_min_error - sum_metrics.clipping_to_max_error
    l0_dropped = -sum_metrics.expected_l0_bounding_error

    # 2. Partition selection (in case of private partition selection).
    # The data which survived contribution bounding is dropped with probability
    # = 1 - partition_keep_probability.
    expected_after_contribution_bounding = sum_metrics.sum - l0_dropped - linf_dropped
    partition_selection_dropped = expected_after_contribution_bounding * (
        1 - partition_keep_probability)

    return metrics.DataDropInfo(l0=l0_dropped,
                                linf=linf_dropped,
                                partition_selection=partition_selection_dropped)


def _create_contribution_bounding_errors(
        sum_metrics: metrics.SumMetrics) -> metrics.ContributionBoundingErrors:
    """Creates ContributionBoundingErrors from per-partition metrics."""
    l0_mean = sum_metrics.expected_l0_bounding_error
    l0_var = sum_metrics.std_l0_bounding_error**2
    l0_mean_var = metrics.MeanVariance(mean=l0_mean, var=l0_var)
    linf_min = sum_metrics.clipping_to_min_error
    linf_max = sum_metrics.clipping_to_max_error
    return metrics.ContributionBoundingErrors(l0=l0_mean_var,
                                              linf_min=linf_min,
                                              linf_max=linf_max)


def _sum_metrics_to_value_error(sum_metrics: metrics.SumMetrics,
                                keep_prob: float,
                                weight: float) -> metrics.ValueErrors:
    """Creates ValueErrors from per-partition metrics."""
    value = sum_metrics.sum
    bounding_errors = _create_contribution_bounding_errors(sum_metrics)
    mean = bounding_errors.l0.mean + bounding_errors.linf_min + bounding_errors.linf_max
    variance = sum_metrics.std_l0_bounding_error**2 + sum_metrics.std_noise**2

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
    if weight != 1:
        # Weight per-partition result for computing weighted sum.
        _multiply_float_dataclasses_field(result,
                                          weight,
                                          fields_to_ignore=["noise_std"])
    return result


def _sum_metrics_to_metric_utility(
        sum_metrics: metrics.SumMetrics, dp_metric: pipeline_dp.Metric,
        partition_keep_probability: float,
        partition_weight: float) -> metrics.MetricUtility:
    """Creates cross-partition MetricUtility from 1 partition utility.

    Attributes:
        sum_metrics: per-partition utility metric.
        dp_metric: metric for which utility is computed (e.g. COUNT)
        partition_keep_probability: partition selection probability.
    """
    data_dropped = _sum_metrics_to_data_dropped(sum_metrics,
                                                partition_keep_probability,
                                                dp_metric)
    absolute_error = _sum_metrics_to_value_error(sum_metrics,
                                                 partition_keep_probability,
                                                 partition_weight)
    relative_error = absolute_error.to_relative(sum_metrics.sum)

    return metrics.MetricUtility(metric=dp_metric,
                                 noise_std=sum_metrics.std_noise,
                                 noise_kind=sum_metrics.noise_kind,
                                 ratio_data_dropped=data_dropped,
                                 absolute_error=absolute_error,
                                 relative_error=relative_error)


def _partition_metrics_public_partitions(
        is_empty_partition: bool) -> metrics.PartitionsInfo:
    result = metrics.PartitionsInfo(public_partitions=True,
                                    num_dataset_partitions=0,
                                    num_non_public_partitions=0,
                                    num_empty_partitions=0)
    if is_empty_partition:
        result.num_empty_partitions = 1
    else:
        result.num_dataset_partitions = 1
    return result


def _partition_metrics_private_partitions(
        prob_keep: float) -> metrics.PartitionsInfo:
    kept_partitions = metrics.MeanVariance(mean=prob_keep,
                                           var=prob_keep * (1 - prob_keep))
    return metrics.PartitionsInfo(public_partitions=False,
                                  num_dataset_partitions=1,
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
    assert type(dataclass1) == type(dataclass2), \
        f"type(dataclass1) = {type(dataclass1)} != type(dataclass2) = {type(dataclass2)}"
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


def _multiply_float_dataclasses_field(dataclass,
                                      factor: float,
                                      fields_to_ignore: List[str] = []) -> None:
    """Recursively multiplies all float fields of the dataclass by given number.

    Warning: it modifies 'dataclass' argument.
    """
    fields = dataclasses.fields(dataclass)
    for field in fields:
        if field.name in fields_to_ignore:
            continue
        value = getattr(dataclass, field.name)
        if value is None:
            continue
        if field.type is float:
            setattr(dataclass, field.name, value * factor)
        elif dataclasses.is_dataclass(value):
            _multiply_float_dataclasses_field(value, factor)


def _per_partition_to_utility_report(
        per_partition_utility: metrics.PerPartitionMetrics,
        dp_metrics: List[pipeline_dp.Metric], public_partitions: bool,
        partition_weight: float) -> metrics.UtilityReport:
    """Converts per-partition metrics to cross-partition utility report."""
    # Fill partition selection metrics.
    if public_partitions:
        prob_to_keep = 1
        is_empty_partition = per_partition_utility.raw_statistics.count == 0
        partition_metrics = _partition_metrics_public_partitions(
            is_empty_partition)
    else:
        prob_to_keep = per_partition_utility.partition_selection_probability_to_keep
        partition_metrics = _partition_metrics_private_partitions(prob_to_keep)
    # Fill metric errors.
    metric_errors = None
    if dp_metrics:
        assert len(per_partition_utility.metric_errors) == len(dp_metrics)
        metric_errors = []
        for metric_error, dp_metric in zip(per_partition_utility.metric_errors,
                                           dp_metrics):
            metric_errors.append(
                _sum_metrics_to_metric_utility(metric_error, dp_metric,
                                               prob_to_keep, partition_weight))

    return metrics.UtilityReport(configuration_index=-1,
                                 partitions_info=partition_metrics,
                                 metric_errors=metric_errors)


def _merge_partition_metrics(metrics1: metrics.PartitionsInfo,
                             metrics2: metrics.PartitionsInfo) -> None:
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
    _merge_partition_metrics(report1.partitions_info, report2.partitions_info)
    if report1.metric_errors is None:
        return
    assert len(report1.metric_errors) == len(report2.metric_errors)
    for utility1, utility2 in zip(report1.metric_errors, report2.metric_errors):
        _merge_metric_utility(utility1, utility2)


def _average_utility_report(report: metrics.UtilityReport, sums_actual: Tuple,
                            total_weight: float) -> None:
    """Averages fields of the 'report' across partitions."""
    if not report.metric_errors:
        return

    for sum_actual, metric_error in zip(sums_actual, report.metric_errors):
        _multiply_float_dataclasses_field(
            metric_error,
            1.0 / total_weight,
            fields_to_ignore=["noise_std", "ratio_data_dropped"])
        scaling_factor = 1 if sum_actual == 0 else 1.0 / sum_actual
        _multiply_float_dataclasses_field(metric_error.ratio_data_dropped,
                                          scaling_factor)


def partition_size_weight_fn(
        per_partition_metrics: metrics.PerPartitionMetrics) -> float:
    """Weights partitions according to their size."""
    # Only one metric is calculated as of now.
    return per_partition_metrics.metric_errors[0].sum


def equal_weight_fn(
        per_partition_metrics: metrics.PerPartitionMetrics) -> float:
    """Weights partitions according to their probability to be kept."""
    # For the public partitions weights will be 1, and we will do normal
    # averaging because total weight will equal to the total number of
    # partitions. The function assumes that
    # partition_selection_probability_to_keep for public partitions is 1 and all
    # public partitions including empty are processed in CrossPartitionCombiner.
    # For private partitions we will do weighted average and
    # total weight will equal to mean number of kept partitions
    # (`partitions.kept_partitions.mean`).
    return per_partition_metrics.partition_selection_probability_to_keep


class CrossPartitionCombiner(pipeline_dp.combiners.Combiner):
    """A combiner for aggregating error metrics across partitions"""
    # Accumulator is a tuple of
    # 1. The sum of non dp metrics, which is used for averaging of error
    # metrics.
    # 2. metrics.UtilityReport contains error metrics.
    # 3. Accumulated weight. Used to calculate total weight after accumulation.
    # During creation of accumulator in `create_accumulator` the initial weight
    # is applied to metric errors of a partition.
    AccumulatorType = Tuple[Tuple, metrics.UtilityReport, float]

    def __init__(self,
                 dp_metrics: List[pipeline_dp.Metric],
                 public_partitions: bool,
                 weight_fn: Callable[[metrics.PerPartitionMetrics],
                                     float] = equal_weight_fn):
        self._dp_metrics = dp_metrics
        self._public_partitions = public_partitions
        self._weight_fn = weight_fn

    def create_accumulator(
            self, metrics: metrics.PerPartitionMetrics) -> AccumulatorType:
        actual_metrics = tuple(me.sum for me in metrics.metric_errors)
        weight = self._weight_fn(metrics)
        return actual_metrics, _per_partition_to_utility_report(
            metrics, self._dp_metrics, self._public_partitions, weight), weight

    def merge_accumulators(self, acc1: AccumulatorType,
                           acc2: AccumulatorType) -> AccumulatorType:
        sum_actual1, report1, weight1 = acc1
        sum_actual2, report2, weight2 = acc2
        sum_actual = tuple(x + y for x, y in zip(sum_actual1, sum_actual2))
        _merge_utility_reports(report1, report2)
        weight = weight1 + weight2
        return sum_actual, report1, weight

    def compute_metrics(self, acc: AccumulatorType) -> metrics.UtilityReport:
        """Returns UtilityReport with final metrics."""
        sum_actual, report, total_weight = acc
        report_copy = copy.deepcopy(report)
        _average_utility_report(report_copy, sum_actual, total_weight)
        return report_copy

    def metrics_names(self):
        return []  # Not used for utility analysis

    def explain_computation(self):
        return None  # Not used for utility analysis
