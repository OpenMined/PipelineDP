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

import copy
import dataclasses
from typing import Iterable, Optional, Union

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
from pipeline_dp import pipeline_backend
import utility_analysis_new
import utility_analysis_new.contribution_bounders as utility_contribution_bounders
import utility_analysis_new.combiners as utility_analysis_combiners


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations.

    This class reuses the computation graph from the DP computation code by
    subclassing from pipeline_dp.DPEngine and by replacing some nodes of the
    computational graph from DP computation to analysis.
    """

    def __init__(self, budget_accountant: budget_accounting.BudgetAccountant,
                 backend: pipeline_backend.PipelineBackend):
        super().__init__(budget_accountant, backend)
        self._is_public_partitions = None

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: pipeline_dp.DataExtractors,
                  public_partitions=None):
        raise ValueError("UtilityAnalysisEngine.aggregate can't be called.\n"
                         "If you like to perform utility analysis use "
                         "UtilityAnalysisEngine.aggregate.\n"
                         "If you like to perform DP computations use "
                         "DPEngine.aggregate.")

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
            from elements of 'col'. In case if the analysis performed on
            pre-aggregated data, it should have type PreAggregateExtractors
            otherwise DataExtractors.
          public_partitions: A collection of partition keys that will be present
            in the result. If not provided, the utility analysis with private
            partition selection will be performed.

        Returns:
          A collection with elements (partition_key, utility analysis metrics).
        """
        _check_utility_analysis_params(options, data_extractors)
        self._options = options
        self._is_public_partitions = public_partitions is not None

        # Build the computation graph from the parent class by calling
        # aggregate().
        result = super().aggregate(col, options.aggregate_params,
                                   data_extractors, public_partitions)

        self._is_public_partitions = None
        self._options = None
        return result

    def preaggregate(self, col, data_extractors: pipeline_dp.DataExtractors,
                     partitions_sampling_prob: float):
        col = super()._extract_columns(col, data_extractors)
        bounder = utility_contribution_bounders.SamplingL0LinfContributionBounder(
            partitions_sampling_prob)
        return bounder.bound_contributions(col,
                                           params=None,
                                           backend=self._backend,
                                           report_generator=None,
                                           aggregate_fn=lambda x: x)

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder for utility analysis."""
        if self._options.pre_aggregated_data:
            return utility_contribution_bounders.NoOpContributionBounder()
        return utility_contribution_bounders.SamplingL0LinfContributionBounder(
            self._options.partitions_sampling_prob)

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
        """Extract columns using data_extractors."""
        if self._options.pre_aggregated_data:
            return self._backend.map(
                col, lambda row: (data_extractors.partition_extractor(row),
                                  data_extractors.preaggregate_extractor(row)),
                "Extract (partition_key, preaggregate_data))")
        return super()._extract_columns(col, data_extractors)

    def _check_aggregate_params(
        self, col, params: pipeline_dp.AggregateParams,
        data_extractors: Union[pipeline_dp.DataExtractors,
                               utility_analysis_new.PreAggregateExtractors]):
        # Do not check data_extractors. The parent implementation does not
        # support PreAggregateExtractors.
        super()._check_aggregate_params(col,
                                        params,
                                        data_extractors=None,
                                        check_data_extractors=False)


def _check_utility_analysis_params(
    options: utility_analysis_new.UtilityAnalysisOptions,
    data_extractors: Union[pipeline_dp.DataExtractors,
                           utility_analysis_new.PreAggregateExtractors]):
    # Check correctness of data extractors.
    if options.pre_aggregated_data:
        if not isinstance(data_extractors,
                          utility_analysis_new.PreAggregateExtractors):
            raise ValueError(
                "options.pre_aggregated_data is set to true but "
                "PreAggregateExtractors aren't provided. PreAggregateExtractors"
                " should be specified for pre-aggregated data.")
    elif not isinstance(data_extractors, pipeline_dp.DataExtractors):
        raise ValueError(
            "pipeline_dp.DataExtractors should be specified for raw data.")

    # Check aggregate_params.
    params = options.aggregate_params
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
