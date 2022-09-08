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
"""DPEngine for utility analysis."""
from typing import List
import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
import utility_analysis_new.contribution_bounders as utility_contribution_bounders
import utility_analysis_new.combiners as utility_analysis_combiners


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations."""

    def __init__(self, budget_accountant: budget_accounting.BudgetAccountant,
                 backend: pipeline_dp.pipeline_backend.PipelineBackend):
        super().__init__(budget_accountant, backend)
        self._is_public_partitions = None

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: pipeline_dp.DataExtractors,
                  public_partitions=None):
        _check_utility_analysis_params(params, public_partitions)
        self._is_public_partitions = public_partitions is not None
        result = super().aggregate(col, params, data_extractors,
                                   public_partitions)
        aggregate_error_combiners = self._create_aggregate_error_compound_combiner(
            params, [0.1, 0.5, 0.9, 0.99])
        self._is_public_partitions = None
        # TODO: Implement combine_accumulators (w/o per_key)
        keyed_by_same_key = self._backend.map(
            result, lambda v: (None, v[1]), "Rekey partitions by the same key")
        accumulators = self._backend.map_values(
            keyed_by_same_key, aggregate_error_combiners.create_accumulator,
            "Create accumulators for aggregating error metrics")
        aggregates = self._backend.combine_accumulators_per_key(
            accumulators, aggregate_error_combiners,
            "Combine aggregate metrics from per-partition error metrics")
        aggregates = self._backend.map_values(
            aggregates, aggregate_error_combiners.compute_metrics,
            "Compute aggregate metrics")
        return self._backend.values(aggregates, "Drop key")

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder for utility analysis."""
        return utility_contribution_bounders.SamplingCrossAndPerPartitionContributionBounder(
        )

    def _create_compound_combiner(
        self, aggregate_params: pipeline_dp.AggregateParams
    ) -> combiners.CompoundCombiner:
        mechanism_type = aggregate_params.noise_kind.convert_to_mechanism_type()
        internal_combiners = []
        if not self._is_public_partitions:
            budget = self._budget_accountant.request_budget(
                mechanism_type=pipeline_dp.MechanismType.GENERIC)
            internal_combiners.append(
                utility_analysis_combiners.PartitionSelectionCombiner(
                    combiners.CombinerParams(budget, aggregate_params)))
        if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
            budget = self._budget_accountant.request_budget(
                mechanism_type, weight=aggregate_params.budget_weight)
            internal_combiners.append(
                utility_analysis_combiners.UtilityAnalysisCountCombiner(
                    combiners.CombinerParams(budget, aggregate_params)))
        if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
            budget = self._budget_accountant.request_budget(
                mechanism_type, weight=aggregate_params.budget_weight)
            internal_combiners.append(
                utility_analysis_combiners.UtilityAnalysisSumCombiner(
                    combiners.CombinerParams(budget, aggregate_params)))
        if pipeline_dp.Metrics.PRIVACY_ID_COUNT in aggregate_params.metrics:
            budget = self._budget_accountant.request_budget(
                mechanism_type, weight=aggregate_params.budget_weight)
            internal_combiners.append(
                utility_analysis_combiners.
                UtilityAnalysisPrivacyIdCountCombiner(
                    combiners.CombinerParams(budget, aggregate_params)))
        return combiners.CompoundCombiner(internal_combiners,
                                          return_named_tuple=False)

    def _create_aggregate_error_compound_combiner(
            self, aggregate_params: pipeline_dp.AggregateParams,
            error_quantiles: List[float]) -> combiners.CompoundCombiner:
        internal_combiners = []
        if not self._is_public_partitions:
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

    def _select_private_partitions_internal(self, col,
                                            max_partitions_contributed: int,
                                            max_rows_per_privacy_id: int):
        return col


def _check_utility_analysis_params(params: pipeline_dp.AggregateParams,
                                   public_partitions=None):
    if params.custom_combiners is not None:
        raise NotImplementedError("custom combiners are not supported")
    if not (set(params.metrics).issubset({
            pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
            pipeline_dp.Metrics.PRIVACY_ID_COUNT
    })):
        not_supported_metrics = list(
            set(params.metrics).difference({
                pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
                pipeline_dp.Metrics.PRIVACY_ID_COUNT
            }))
        raise NotImplementedError(
            f"unsupported metric in metrics={not_supported_metrics}")
    if params.contribution_bounds_already_enforced:
        raise NotImplementedError(
            "utility analysis when contribution bounds are already enforced is not supported"
        )
