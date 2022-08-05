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

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
import utility_analysis_new.contribution_bounders as utility_contribution_bounders
import utility_analysis_new.combiners as utility_analysis_combiners


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations."""

    def __init__(self, budget_accountant: 'BudgetAccountant',
                 backend: 'PipelineBackend'):
        super().__init__(budget_accountant, backend)

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: pipeline_dp.DataExtractors,
                  public_partitions=None):
        _check_utility_analysis_params(params, public_partitions)
        metrics = super().aggregate(col, params, data_extractors,
                                    public_partitions)
        # TODO: call utility_analysis_combiners.CountUtilityAnalysisErrorAggregator
        return self._backend.map_values(
            metrics, _compute_high_level_metrics,
            "Compute high-level metrics from variance and expectation")

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
        budget = self._budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        return combiners.CompoundCombiner([
            utility_analysis_combiners.UtilityAnalysisCountCombiner(
                combiners.CombinerParams(budget, aggregate_params))
        ],
                                          return_named_tuple=False)


def _check_utility_analysis_params(params: pipeline_dp.AggregateParams,
                                   public_partitions=None):
    if params.custom_combiners is not None:
        raise NotImplementedError("custom combiners are not supported")
    if params.metrics != [pipeline_dp.Metrics.COUNT]:
        raise NotImplementedError(
            f"supported only count metrics, metrics={params.metrics}")
    if public_partitions is None:
        raise NotImplementedError("only public partitions supported")
    if params.contribution_bounds_already_enforced:
        raise NotImplementedError(
            "utility analysis when contribution bounds are already enforced is not supported"
        )


def _compute_high_level_metrics(
        metrics: utility_analysis_combiners.CountUtilityAnalysisMetrics):
    # Absolute error metrics
    metrics.abs_error_expected = metrics.per_partition_error + metrics.expected_cross_partition_error
    metrics.abs_error_variance = metrics.std_cross_partition_error**2 + metrics.std_noise**2
    # TODO: Implement 99% error

    # Relative error metrics
    metrics.rel_error_expected = metrics.abs_error_expected / metrics.count
    metrics.rel_error_variance = metrics.abs_error_variance / (metrics.count**2)

    return metrics
