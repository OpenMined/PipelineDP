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
from dataclasses import dataclass
from typing import List, Optional

import pipeline_dp
from pipeline_dp import combiners
from pipeline_dp import pipeline_backend
from pipeline_dp import input_validators
from utility_analysis_new import dp_engine
from utility_analysis_new import metrics
import utility_analysis_new.combiners as utility_analysis_combiners


@dataclass
class UtilityAnalysisOptions:
    """Options for the utility analysis."""
    epsilon: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams
    multi_param_configuration: Optional[
        dp_engine.MultiParameterConfiguration] = None

    def __post_init__(self):
        input_validators.validate_epsilon_delta(self.epsilon, self.delta,
                                                "UtilityAnalysisOptions")

    @property
    def n_parameters(self):
        if self.multi_param_configuration is None:
            return 1
        return self.multi_param_configuration.size


def perform_utility_analysis(col,
                             backend: pipeline_backend.PipelineBackend,
                             options: UtilityAnalysisOptions,
                             data_extractors: pipeline_dp.DataExtractors,
                             public_partitions=None):
    """Performs utility analysis for DP aggregations.

    Args:
      col: collection where all elements are of the same type.
      backend: PipelineBackend with which the utility analysis will be run.
      options: options for utility analysis.
      data_extractors: functions that extract needed pieces of information
            from elements of 'col'.
      public_partitions: A collection of partition keys that will be present
            in the result. If not provided, the utility analysis with private
            partition selection will be performed.
    Returns:
      1 element collection which contains utility analysis metrics.
    """
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=options.epsilon, total_delta=options.delta)
    engine = dp_engine.UtilityAnalysisEngine(
        budget_accountant=budget_accountant, backend=backend)
    result = engine.aggregate(
        col,
        params=options.aggregate_params,
        data_extractors=data_extractors,
        public_partitions=public_partitions,
        multi_param_configuration=options.multi_param_configuration)
    budget_accountant.compute_budgets()
    # result : (partition_key, per_partition_metrics)

    aggregate_error_combiners = _create_aggregate_error_compound_combiner(
        options.aggregate_params, [0.1, 0.5, 0.9, 0.99], public_partitions,
        options.n_parameters)
    # TODO: Implement combine_accumulators (w/o per_key)
    keyed_by_same_key = backend.map(result, lambda v: (None, v[1]),
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

    # accumulators : (aggregate_metrics)

    def pack_metrics(aggregate_metrics) -> List[metrics.AggregateMetrics]:
        # aggregate_metrics is a flat list of PartitionSelectionMetrics and
        # AggregateErrorMetrics with options.n_parameters sequential
        # configurations of metrics. Each AggregateErrorMetrics within a
        # configuration correspond to a different aggregation.
        metrics_per_config = len(aggregate_metrics) // options.n_parameters
        return_list = []
        for i in range(0, options.n_parameters):
            packed_metrics = metrics.AggregateMetrics()
            for j in range(i * metrics_per_config,
                           (i + 1) * metrics_per_config):
                _populate_packed_metrics(packed_metrics, aggregate_metrics[j])
            return_list.append(packed_metrics)
        return return_list

    return backend.map(aggregates, pack_metrics,
                       "Pack metrics from the same run")
    # (aggregate_metrics)


# Sets the appropriate field of packed_metrics with aggregate_error_metric
# according to the type of aggregate_error_metric.
def _populate_packed_metrics(packed_metrics: metrics.AggregateMetrics,
                             aggregate_error_metric):
    if isinstance(aggregate_error_metric, metrics.PartitionSelectionMetrics):
        packed_metrics.partition_selection_metrics = aggregate_error_metric
    elif aggregate_error_metric.metric_type == metrics.AggregateMetricType.PRIVACY_ID_COUNT:
        packed_metrics.privacy_id_count_metrics = aggregate_error_metric
    elif aggregate_error_metric.metric_type == metrics.AggregateMetricType.COUNT:
        packed_metrics.count_metrics = aggregate_error_metric


def _create_aggregate_error_compound_combiner(
        aggregate_params: pipeline_dp.AggregateParams,
        error_quantiles: List[float], public_partitions: bool,
        n_parameters: int) -> combiners.CompoundCombiner:
    internal_combiners = []
    for i in range(n_parameters):
        if not public_partitions:
            internal_combiners.append(
                utility_analysis_combiners.
                PrivatePartitionSelectionAggregateErrorMetricsCombiner(
                    None, error_quantiles))
        # WARNING: The order here needs to follow the order in
        # UtilityAnalysisEngine._create_compound_combiner().
        if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
            internal_combiners.append(
                utility_analysis_combiners.CountAggregateErrorMetricsCombiner(
                    None, error_quantiles))
        if pipeline_dp.Metrics.PRIVACY_ID_COUNT in aggregate_params.metrics:
            internal_combiners.append(
                utility_analysis_combiners.
                PrivacyIdCountAggregateErrorMetricsCombiner(
                    None, error_quantiles))
    return utility_analysis_combiners.AggregateErrorMetricsCompoundCombiner(
        internal_combiners, return_named_tuple=False)
