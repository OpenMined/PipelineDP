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
from pipeline_dp import pipeline_backend
import utility_analysis_new.contribution_bounders as utility_contribution_bounders
import utility_analysis_new.combiners as utility_analysis_combiners
from dataclasses import dataclass
from typing import Sequence


@dataclass
class CountMultiRunConfiguration:
    max_partitions_contributed: Sequence[int]
    max_contributions_per_partition: Sequence[int]

    @property
    def num_configurations(self):
        return len(self.max_partitions_contributed)

    def get_configuration(self, params: pipeline_dp.AggregateParams,
                          index: int):
        # other name?
        params = copy.copy(params)
        params.max_partitions_contributed = self.max_partitions_contributed[
            index]
        params.max_contributions_per_partition = self.max_contributions_per_partition[
            index]
        return params

    # min_value: Optional[float] = None
    # max_value: Optional[float] = None
    # min_sum_per_partition: Optional[float] = None
    # max_sum_per_partition: Optional[float] = None

    def __post_init__(self):
        if len(self.max_partitions_contributed) != len(
                self.max_contributions_per_partition):
            raise ValueError(
                'CountMultiRunConfiguration: max_partitions_contributed and '
                'max_contributions_per_partition must be the same length.')


@dataclass
class UtilityAnalysisAggregateParams(pipeline_dp.AggregateParams):
    count_multi_run: CountMultiRunConfiguration = None


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations."""

    def __init__(self, budget_accountant: budget_accounting.BudgetAccountant,
                 backend: pipeline_backend.PipelineBackend):
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

        self._is_public_partitions = None
        return result

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
            if aggregate_params.count_multi_run:
                multi_run = aggregate_params.count_multi_run
                for i in range(multi_run.num_configurations):
                    params = multi_run.get_configuration(aggregate_params, i)
                    internal_combiners.append(
                        utility_analysis_combiners.UtilityAnalysisCountCombiner(
                            combiners.CombinerParams(budget, params)))
            else:
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
        return utility_analysis_combiners.UtilityAnalysisCompoundCombiner(
            internal_combiners, return_named_tuple=False)

    def _select_private_partitions_internal(
            self, col, max_partitions_contributed: int,
            max_rows_per_privacy_id: int,
            strategy: pipeline_dp.PartitionSelectionStrategy):
        # Utility analysis of private partition selection is performed in a
        # corresponding combiners (unlike actual DP computations). So this
        # function is no-op.
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
            "utility analysis when contribution bounds are already enforced is "
            "not supported")
