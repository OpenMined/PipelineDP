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
from pipeline_dp import pipeline_backend
import analysis
from analysis import metrics
from analysis import utility_analysis_engine
from analysis import cross_partition_combiners
import copy


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

    col = backend.flat_map(
        col, lambda metrics: ((i, metric) for i, metric in enumerate(metrics)),
        "Unnest metrics")
    # (configuration_index, metrics.PerPartitionMetrics)

    combiner = cross_partition_combiners.CrossPartitionCombiner(
        options.aggregate_params.metrics, public_partitions is not None)

    accumulators = backend.map_values(col, combiner.create_accumulator,
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
        copy_report = copy.deepcopy(report)
        copy_report.configuration_index = index
        return copy_report

    result = backend.map_tuple(cross_partition_metrics, add_index, "Add index")
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
