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
from typing import Iterable, Union

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
from pipeline_dp import pipeline_backend
import utility_analysis_new
import utility_analysis_new.contribution_bounders as utility_contribution_bounders
import utility_analysis_new.combiners as utility_analysis_combiners


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations."""

    def __init__(self, budget_accountant: budget_accounting.BudgetAccountant,
                 backend: pipeline_backend.PipelineBackend):
        super().__init__(budget_accountant, backend)
        self._is_public_partitions = None
        self._sampling_probability = 1.0

    def analyze(
            self,
            col,
            options: utility_analysis_new.UtilityAnalysisOptions,
            data_extractors: Union[pipeline_dp.DataExtractors,
                                   utility_analysis_new.PreAggregateExtractors],
            public_partitions=None):
        """Performs utility analysis for DP aggregations per partition.

        Args:
          col: collection where all elements are of the same type.
          options: options for utility analysis.
          data_extractors: functions that extract needed pieces of information
            from elements of 'col'.
          public_partitions: A collection of partition keys that will be present
            in the result. If not provided, the utility analysis with private
            partition selection will be performed.

        Returns:
            A collection with elements (pk, utility analysis metrics).
        """
        _check_utility_analysis_params(options.aggregate_params,
                                       public_partitions)  # todo
        self._is_public_partitions = public_partitions is not None
        self._options = options
        result = super().aggregate(col, options.aggregate_params,
                                   data_extractors, public_partitions)
        self._is_public_partitions = None
        self._options = None
        return result

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder for utility analysis."""
        if self._options.pre_aggregated_data:
            return utility_contribution_bounders.NoOpContributionBounder()
        return utility_contribution_bounders.SamplingL0LinfContributionBounder(
            self._sampling_probability)

    def _create_compound_combiner(
        self, aggregate_params: pipeline_dp.AggregateParams
    ) -> combiners.CompoundCombiner:
        mechanism_type = aggregate_params.noise_kind.convert_to_mechanism_type()
        internal_combiners = []
        for params in self._get_aggregate_params(aggregate_params):
            if not self._is_public_partitions:
                budget = self._budget_accountant.request_budget(
                    mechanism_type=pipeline_dp.MechanismType.GENERIC)
                internal_combiners.append(
                    utility_analysis_combiners.PartitionSelectionCombiner(
                        combiners.CombinerParams(budget, params)))
            if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
                budget = self._budget_accountant.request_budget(
                    mechanism_type, weight=aggregate_params.budget_weight)
                internal_combiners.append(
                    utility_analysis_combiners.CountCombiner(
                        combiners.CombinerParams(budget, params)))
            if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
                budget = self._budget_accountant.request_budget(
                    mechanism_type, weight=aggregate_params.budget_weight)
                internal_combiners.append(
                    utility_analysis_combiners.SumCombiner(
                        combiners.CombinerParams(budget, params)))
            if pipeline_dp.Metrics.PRIVACY_ID_COUNT in aggregate_params.metrics:
                budget = self._budget_accountant.request_budget(
                    mechanism_type, weight=aggregate_params.budget_weight)
                internal_combiners.append(
                    utility_analysis_combiners.PrivacyIdCountCombiner(
                        combiners.CombinerParams(budget, params)))
        return utility_analysis_combiners.CompoundCombiner(
            internal_combiners, return_named_tuple=False)

    def _get_aggregate_params(
        self, params: pipeline_dp.AggregateParams
    ) -> Iterable[pipeline_dp.AggregateParams]:
        multi_run_configuration = self._options.multi_param_configuration
        if multi_run_configuration is None:
            yield params
        else:
            for i in range(multi_run_configuration.size):
                yield multi_run_configuration.get_aggregate_params(params, i)

    def _select_private_partitions_internal(
            self, col, max_partitions_contributed: int,
            max_rows_per_privacy_id: int,
            strategy: pipeline_dp.PartitionSelectionStrategy):
        # Utility analysis of private partition selection is performed in a
        # corresponding combiners (unlike actual DP computations). So this
        # function is no-op.
        return col

    def _extract_columns(
        self, col,
        data_extractors: Union[pipeline_dp.DataExtractors,
                               utility_analysis_new.PreAggregateExtractors]):
        """todo"""
        if self._options.pre_aggregated_data:
            # todo: check data_extractors is PreAggregateExtractors
            return self._backend.map(
                col, lambda row: (data_extractors.partition_extractor(row),
                                  data_extractors.preaggregate_extractor(row)),
                "Extract (partition_key, preaggregate_data))")
        return super()._extract_columns(col, data_extractors)


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
