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
from typing import List

import pipeline_dp
from pipeline_dp import combiners
from pipeline_dp import pipeline_backend
from utility_analysis_new import dp_engine
import utility_analysis_new.combiners as utility_analysis_combiners
from dataclasses import dataclass


@dataclass
class UtilityAnalysisOptions:
    """Options for the utility analysis."""
    eps: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams


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
  """
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        total_epsilon=options.eps, total_delta=options.delta)
    engine = dp_engine.UtilityAnalysisEngine(
        budget_accountant=budget_accountant, backend=backend)
    result = engine.aggregate(col,
                              params=options.aggregate_params,
                              data_extractors=data_extractors,
                              public_partitions=public_partitions)
    budget_accountant.compute_budgets()

    def create_aggregate_error_compound_combiner(
            aggregate_params: pipeline_dp.AggregateParams,
            error_quantiles: List[float],
            public_partitions: bool) -> combiners.CompoundCombiner:
        internal_combiners = []
        if not public_partitions:
            internal_combiners.append(
                utility_analysis_combiners.
                PrivatePartitionSelectionAggregateErrorMetricsCombiner(
                    None, error_quantiles))
        if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
            internal_combiners.append(
                utility_analysis_combiners.CountAggregateErrorMetricsCombiner(
                    None, error_quantiles))
        return utility_analysis_combiners.AggregateErrorMetricsCompoundCombiner(
            internal_combiners, return_named_tuple=False)

    aggregate_error_combiners = create_aggregate_error_compound_combiner(
        options.aggregate_params, [0.1, 0.5, 0.9, 0.99],
        public_partitions != None)
    # TODO: Implement combine_accumulators (w/o per_key)
    keyed_by_same_key = backend.map(result, lambda v: (None, v[1]),
                                    "Rekey partitions by the same key")
    accumulators = backend.map_values(
        keyed_by_same_key, aggregate_error_combiners.create_accumulator,
        "Create accumulators for aggregating error metrics")
    aggregates = backend.combine_accumulators_per_key(
        accumulators, aggregate_error_combiners,
        "Combine aggregate metrics from per-partition error metrics")
    aggregates = backend.map_values(aggregates,
                                    aggregate_error_combiners.compute_metrics,
                                    "Compute aggregate metrics")
    return backend.values(aggregates, "Drop key")
