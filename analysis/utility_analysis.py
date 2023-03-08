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
from typing import Any, List, Tuple, Union

import pipeline_dp
from pipeline_dp import combiners
from pipeline_dp import pipeline_backend
import analysis
from analysis import data_structures
from analysis import metrics
from analysis import utility_analysis_engine
import analysis.combiners as utility_analysis_combiners
from analysis import cross_partition_combiners


def perform_utility_analysis(
        col,
        backend: pipeline_backend.PipelineBackend,
        options: analysis.UtilityAnalysisOptions,
        data_extractors: Union[pipeline_dp.DataExtractors,
                               analysis.PreAggregateExtractors],
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
      return_per_partition: if true, it returns tuple, with the 2nd element
        utility analysis per partitions.
    Returns:
         if return_per_partition == False:
            returns 1 element collection which contains TuneResult
        else returns tuple (1 element collection which contains TuneResult,
        a collection which contains utility analysis results per partition).
    """
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=options.epsilon, total_delta=options.delta)
    engine = utility_analysis_engine.UtilityAnalysisEngine(
        budget_accountant=budget_accountant, backend=backend)
    per_partition_analysis_result = engine.analyze(
        col,
        options=options,
        data_extractors=data_extractors,
        public_partitions=public_partitions)
    budget_accountant.compute_budgets()
    # per_partition_analysis_result : (partition_key, per_partition_metrics)
    per_partition_analysis_result = backend.to_multi_transformable_collection(
        per_partition_analysis_result)

    aggregate_error_combiners = _create_aggregate_error_compound_combiner(
        options.aggregate_params, [0.1, 0.5, 0.9, 0.99], public_partitions,
        options.n_configurations)
    # TODO: Implement combine_accumulators (w/o per_key)
    keyed_by_same_key = backend.map(per_partition_analysis_result, lambda v:
                                    (None, v[1]),
                                    "Rekey partitions by the same key")
    # keyed_by_same_key : (None, per_partition_metrics)
    accumulators = backend.map_values(
        keyed_by_same_key, aggregate_error_combiners.create_accumulator,
        "Create accumulators for aggregating error metrics")
    # accumulators : (None, (accumulator))
    aggregates = backend.combine_accumulators_per_key(
        accumulators, aggregate_error_combiners,
        "Combine aggregate metrics from per-partition error metrics")
    # aggregates : (None, (accumulator))

    aggregates = backend.values(aggregates, "Drop key")
    # aggregates: (accumulator)

    aggregates = backend.map(aggregates,
                             aggregate_error_combiners.compute_metrics,
                             "Compute aggregate metrics")

    # aggregates : (aggregate_metrics)

    def pack_metrics(aggregate_metrics) -> List[metrics.AggregateMetrics]:
        # aggregate_metrics is a flat list of PartitionSelectionMetrics and
        # AggregateErrorMetrics with options.n_configurations sequential
        # configurations of metrics. Each AggregateErrorMetrics within a
        # configuration correspond to a different aggregation.
        aggregate_params = list(data_structures.get_aggregate_params(options))
        n_configurations = len(aggregate_params)
        metrics_per_config = len(aggregate_metrics) // n_configurations

        return_list = []
        for i, aggregate_params in enumerate(aggregate_params):
            packed_metrics = metrics.AggregateMetrics(
                input_aggregate_params=aggregate_params)
            for j in range(i * metrics_per_config,
                           (i + 1) * metrics_per_config):
                _populate_packed_metrics(packed_metrics, aggregate_metrics[j])
            return_list.append(packed_metrics)
        return return_list

    result = backend.map(aggregates, pack_metrics,
                         "Pack metrics from the same run")
    # result: (aggregate_metrics)
    if return_per_partition:
        return result, per_partition_analysis_result
    return result


def perform_utility_analysis_new(
        col,
        backend: pipeline_backend.PipelineBackend,
        options: analysis.UtilityAnalysisOptions,
        data_extractors: Union[pipeline_dp.DataExtractors,
                               analysis.PreAggregateExtractors],
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
      return_per_partition: if true, it returns tuple, with the 2nd element
        utility analysis per partitions.
    Returns:
         if return_per_partition == False:
            returns 1 element collection which contains TuneResult
        else returns tuple (1 element collection which contains TuneResult,
        a collection which contains utility analysis results per partition).
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

    per_partition_result = backend.to_multi_transformable_collection(
        per_partition_result)

    per_partition_result = backend.values(per_partition_result,
                                          "Drop partition key")
    # (utility results)

    n_configurations = options.n_configurations
    per_partition_result = backend.flat_map(
        per_partition_result,
        lambda value: _pack_per_partition_metrics(value, n_configurations),
        "Pack per-partition metrics.")
    # (configuration_index, metrics.PerPartitionMetrics)

    combiner = cross_partition_combiners.CrossPartitionCombiner(
        options.aggregate_params.metrics, public_partitions is not None)

    accumulators = backend.map_values(per_partition_result,
                                      combiner.create_accumulator,
                                      "Create accumulators")
    # accumulators : (configuration_index, accumulator)

    accumulators = backend.combine_accumulators_per_key(
        accumulators, combiner, "Combine cross-partition metrics")
    # accumulators : (configuration_index, accumulator)

    cross_partition_metrics = backend.map_values(
        accumulators, combiner.compute_metrics,
        "Compute cross-partition metrics")

    # cross_partition_metrics : (configuration_index, UtilityReport)
    def add_index(index,
                  report: metrics.UtilityReport) -> metrics.UtilityReport:
        report.configuration_index = index
        return report

    return backend.map_tuple(cross_partition_metrics, add_index, "Add index")


def _pack_per_partition_metrics(
        utility_result: List[Any],
        n_configurations: int) -> List[Tuple[int, metrics.PerPartitionMetrics]]:
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
    empty_per_partition_metric = lambda: metrics.PerPartitionMetrics(1, [])
    result = [(i, empty_per_partition_metric()) for i in range(n_configurations)
             ]

    # Fill 'result' from 'utility_metrics'.
    for i, metric in enumerate(utility_result):
        i_configuration = i // n_metrics
        if isinstance(metric, float):  # partition selection
            result[i_configuration][
                1].partition_selection_probability_to_keep = metric
        else:
            result[i_configuration][1].metric_errors.append(metric)
    return result


def _populate_packed_metrics(packed_metrics: metrics.AggregateMetrics, metric):
    """Sets the appropriate packed_metrics field with 'metric' according to 'metric' type."""
    if isinstance(metric, metrics.PartitionSelectionMetrics):
        packed_metrics.partition_selection_metrics = metric
    elif metric.metric_type == metrics.AggregateMetricType.PRIVACY_ID_COUNT:
        packed_metrics.privacy_id_count_metrics = metric
    elif metric.metric_type == metrics.AggregateMetricType.COUNT:
        packed_metrics.count_metrics = metric
    elif metric.metric_type == metrics.AggregateMetricType.SUM:
        packed_metrics.sum_metrics = metric


def _create_aggregate_error_compound_combiner(
        aggregate_params: pipeline_dp.AggregateParams,
        error_quantiles: List[float], public_partitions: bool,
        n_configurations: int) -> combiners.CompoundCombiner:
    internal_combiners = []
    for i in range(n_configurations):
        if not public_partitions:
            internal_combiners.append(
                utility_analysis_combiners.
                PrivatePartitionSelectionAggregateErrorMetricsCombiner(
                    error_quantiles))
        # WARNING: The order here needs to follow the order in
        # UtilityAnalysisEngine._create_compound_combiner().
        if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
            internal_combiners.append(
                utility_analysis_combiners.SumAggregateErrorMetricsCombiner(
                    metrics.AggregateMetricType.SUM, error_quantiles))
        if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
            internal_combiners.append(
                utility_analysis_combiners.SumAggregateErrorMetricsCombiner(
                    metrics.AggregateMetricType.COUNT, error_quantiles))
        if pipeline_dp.Metrics.PRIVACY_ID_COUNT in aggregate_params.metrics:
            internal_combiners.append(
                utility_analysis_combiners.SumAggregateErrorMetricsCombiner(
                    metrics.AggregateMetricType.PRIVACY_ID_COUNT,
                    error_quantiles))
    return utility_analysis_combiners.AggregateErrorMetricsCompoundCombiner(
        internal_combiners, return_named_tuple=False)
