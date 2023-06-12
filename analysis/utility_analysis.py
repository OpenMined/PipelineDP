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
"""Public API for performing utility analysis."""
from typing import Any, Iterable, List, Tuple, Union

import pipeline_dp
from pipeline_dp import pipeline_backend
import analysis
from analysis import metrics
from analysis import utility_analysis_engine
from analysis import cross_partition_combiners
import copy
import bisect


def _generate_bucket_bounds():
    result = [0, 1]
    for i in range(1, 10):
        result.append(10**i)
        result.append(2 * 10**i)
        result.append(5 * 10**i)
    return tuple(result)


# Bucket bounds for metrics. UtilityReportHistogram.
# Bounds are logarithmic: [0, 1] + [1, 2, 5]*10**i , for i = 1, 10.
BUCKET_BOUNDS = _generate_bucket_bounds()


def perform_utility_analysis(
        col,
        backend: pipeline_backend.PipelineBackend,
        options: analysis.UtilityAnalysisOptions,
        data_extractors: Union[pipeline_dp.DataExtractors,
                               pipeline_dp.PreAggregateExtractors],
        public_partitions=None,
        return_per_partition: bool = False):
    """Performs utility analysis for DP aggregations.

    Args:
      col: collection where all elements are of the same type.
      backend: PipelineBackend with which the utility analysis will be run.
      options: options for utility analysis.
      data_extractors: functions that extract needed pieces of information
        from elements of 'col'. In case if the analysis performed on
        pre-aggregated data, it should have type PreAggregateExtractors
        otherwise DataExtractors.
      public_partitions: A collection of partition keys that will be present
        in the result. If not provided, the utility analysis with private
        partition selection will be performed.
      return_per_partition: if true, in addition it returns utility analysis per
        partitions as a 2nd element.
    Returns:
         if return_per_partition == False:
            returns collections which contains metrics.UtilityReport with
            one report per each input configuration
         else:
            returns a tuple, with the 1st element as in first 'if' clause and
            the 2nd a collection with elements
            (partition_key, [metrics.PerPartitionMetrics]).
    """
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=options.epsilon, total_delta=options.delta)
    engine = utility_analysis_engine.UtilityAnalysisEngine(
        budget_accountant=budget_accountant, backend=backend)
    per_partition_result = engine.analyze(col,
                                          options=options,
                                          data_extractors=data_extractors,
                                          public_partitions=public_partitions)
    # (partition_key, utility results)
    budget_accountant.compute_budgets()

    n_configurations = options.n_configurations
    per_partition_result = backend.map_values(
        per_partition_result,
        lambda value: _pack_per_partition_metrics(value, n_configurations),
        "Pack per-partition metrics.")
    # (partition_key, (metrics.PerPartitionMetrics))

    per_partition_result = backend.to_multi_transformable_collection(
        per_partition_result)
    # (partition_key, (metrics.PerPartitionMetrics))

    col = backend.values(per_partition_result, "Drop partition key")
    # ((metrics.PerPartitionMetrics))

    col = backend.flat_map(col, _unnest_metrics, "Unnest metrics")
    # ((configuration_index, bucket), metrics.PerPartitionMetrics)

    combiner = cross_partition_combiners.CrossPartitionCombiner(
        options.aggregate_params.metrics, public_partitions is not None)

    accumulators = backend.map_values(col, combiner.create_accumulator,
                                      "Create accumulators")
    # accumulators : ((configuration_index, bucket), accumulator)

    accumulators = backend.combine_accumulators_per_key(
        accumulators, combiner, "Combine cross-partition metrics")
    # accumulators : ((configuration_index, bucket), accumulator)

    cross_partition_metrics = backend.map_values(
        accumulators, combiner.compute_metrics,
        "Compute cross-partition metrics")
    #  ((configuration_index, bucket), UtilityReport)

    cross_partition_metrics = backend.map_tuple(
        cross_partition_metrics, lambda key, value: (key[0], (key[1], value)),
        "Rekey")
    # (configuration_index, (bucket, UtilityReport))

    cross_partition_metrics = backend.group_by_key(cross_partition_metrics,
                                                   "Group by configuration")
    # (configuration_index, Iterable[(bucket, UtilityReport)])
    result = backend.map_tuple(cross_partition_metrics, _group_utility_reports,
                               "Group utility reports")
    # result: (UtilityReport)

    if return_per_partition:
        return result, per_partition_result
    return result


def _pack_per_partition_metrics(
        utility_result: List[Any],
        n_configurations: int) -> Tuple[metrics.PerPartitionMetrics]:
    """Packs per-partition metrics.

    Arguments:
        utility_result: a list with per-partition results, which contains
          results for all configurations.
        n_configurations: the number of configuration of parameters for which
          the utility analysis is computed.

    Returns: a list of element (i_configuration, PerPartitionMetrics),
    where each element corresponds to one of the configuration of the input
    parameters.
    """
    n_metrics = len(utility_result) // n_configurations

    # Create 'result' with empty elements.
    empty_partition_metric = lambda: metrics.PerPartitionMetrics(1, [])
    result = tuple(empty_partition_metric() for _ in range(n_configurations))

    # Fill 'result' from 'utility_metrics'.
    for i, metric in enumerate(utility_result):
        i_configuration = i // n_metrics
        ith_result = result[i_configuration]
        if isinstance(metric, float):  # partition selection
            ith_result.partition_selection_probability_to_keep = metric
        else:
            ith_result.metric_errors.append(metric)
    return result


def _get_lower_bound(n: int) -> int:
    if n < 0:
        return 0
    return BUCKET_BOUNDS[bisect.bisect_right(BUCKET_BOUNDS, n) - 1]


def _get_upper_bound(n: int) -> int:
    if n < 0:
        return 0
    index = bisect.bisect_right(BUCKET_BOUNDS, n)
    if index > len(BUCKET_BOUNDS):
        return -1  # todo
    return BUCKET_BOUNDS[index]


def _unnest_metrics(
    metrics: List[metrics.PerPartitionMetrics]
) -> Iterable[Tuple[Any, metrics.PerPartitionMetrics]]:
    """Unnest metrics from different configurations."""
    for i, metric in enumerate(metrics):
        yield ((i, None), metric)
    if metrics[0].metric_errors:
        # Emits metrics for computing histogram by partition size.
        actual_bucket_value = metrics[0].metric_errors[0].sum
        bucket = _get_lower_bound(actual_bucket_value)
        for i, metric in enumerate(metrics):
            yield ((i, bucket), metric)


def _group_utility_reports(
        configuration_index: int,
        reports: List[metrics.UtilityReport]) -> metrics.UtilityReport:
    """Groups utility reports for one configuration.

    'reports' contains the global report, i.e. which corresponds to all
    partitions and reports that corresponds to partitions of some size range.
    This function creates UtilityReportHistogram from reports by size range and
    sets it to the global report.
    """
    global_report = None
    histogram_reports = []
    for lower_bucket_bound, report in reports:
        #  Apache Beam does not allow input data to be changed during a map
        #  stage. So 'report' has to be copied.
        report = copy.deepcopy(report)
        report.configuration_index = configuration_index
        if lower_bucket_bound is None:
            # only the report which corresponds to all partitions does not have
            # the bucket.
            global_report = report
        else:
            histogram_reports.append((lower_bucket_bound, report))
    if global_report is None:
        # it should not happen, but it better to process gracefully in case
        # if it happens and return None.
        return None

    if not histogram_reports:
        # It happens in SelectPartitions case.
        # TODO(dvadym): cover SelectPartitions case as well.
        return global_report
    histogram_reports.sort()  # sort by lower_bucket_bound
    utility_report_histogram = [
        metrics.UtilityReportBin(lower_bound, _get_upper_bound(lower_bound),
                                 report)
        for lower_bound, report in histogram_reports
    ]
    global_report.utility_report_histogram = utility_report_histogram
    return global_report
