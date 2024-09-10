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
from typing import Optional, Union

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
from pipeline_dp import pipeline_backend
import analysis
import analysis.contribution_bounders as utility_contribution_bounders
from analysis import per_partition_combiners
from analysis import data_structures


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

    def analyze(self,
                col,
                options: analysis.UtilityAnalysisOptions,
                data_extractors: Union[pipeline_dp.DataExtractors,
                                       pipeline_dp.PreAggregateExtractors],
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
        self._add_report_generator(options.aggregate_params, "analyze")
        result = super()._aggregate(col, options.aggregate_params,
                                    data_extractors, public_partitions)

        self._is_public_partitions = None
        self._options = None
        return result

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams,
        expects_per_partition_sampling: bool
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder for utility analysis."""
        if self._options.pre_aggregated_data:
            return utility_contribution_bounders.NoOpContributionBounder()
        return utility_contribution_bounders.AnalysisContributionBounder(
            self._options.partitions_sampling_prob)

    def _create_compound_combiner(
        self, aggregate_params: pipeline_dp.AggregateParams
    ) -> combiners.CompoundCombiner:
        # Create Utility analysis combiners.
        internal_combiners = [per_partition_combiners.RawStatisticsCombiner()]
        for params, min_max_sum_per_partition in data_structures.get_aggregate_params(
                self._options):
            # Each parameter configuration has own BudgetAccountant which allows
            # different mechanisms to be used in different configurations.
            budget_accountant = copy.deepcopy(self._budget_accountant)

            mechanism_type = None
            if params.noise_kind is None:
                # This is select partition case.
                assert not aggregate_params.metrics, \
                    f"Noise kind should be given when " \
                    f"{aggregate_params.metrics[0]} is analyzed"
            else:
                mechanism_type = params.noise_kind.convert_to_mechanism_type()
            # WARNING: Do not change the order here,
            # _create_aggregate_error_compound_combiner() in utility_analysis.py
            # depends on it.
            if not self._is_public_partitions:
                internal_combiners.append(
                    per_partition_combiners.PartitionSelectionCombiner(
                        budget_accountant.request_budget(
                            pipeline_dp.MechanismType.GENERIC), params))
            if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
                for i_column, (min_sum,
                               max_sum) in enumerate(min_max_sum_per_partition):
                    sum_params = copy.deepcopy(params)
                    sum_params.min_sum_per_partition = min_sum
                    sum_params.max_sum_per_partition = max_sum
                    internal_combiners.append(
                        per_partition_combiners.SumCombiner(
                            budget_accountant.request_budget(mechanism_type),
                            params,
                            i_column=i_column))
            if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
                internal_combiners.append(
                    per_partition_combiners.CountCombiner(
                        budget_accountant.request_budget(mechanism_type),
                        copy.deepcopy(params)))
            if pipeline_dp.Metrics.PRIVACY_ID_COUNT in aggregate_params.metrics:
                internal_combiners.append(
                    per_partition_combiners.PrivacyIdCountCombiner(
                        budget_accountant.request_budget(mechanism_type),
                        copy.deepcopy(params)))
            budget_accountant.compute_budgets()

        return per_partition_combiners.CompoundCombiner(
            internal_combiners, return_named_tuple=False)

    def _select_private_partitions_internal(
            self, col, max_partitions_contributed: int,
            max_rows_per_privacy_id: int,
            strategy: pipeline_dp.PartitionSelectionStrategy,
            pre_threshold: Optional[int]):
        # Utility analysis of private partition selection is performed in a
        # corresponding combiners (unlike actual DP computations). So this
        # function is no-op.
        return col

    def _extract_columns(
        self, col, data_extractors: Union[pipeline_dp.DataExtractors,
                                          pipeline_dp.PreAggregateExtractors]):
        """Extract columns using data_extractors."""
        if self._options.pre_aggregated_data:
            # The output elements format (privacy_id, partition_key, value).
            # For pre-aggregation privacy_id is not needed. So None is return
            # as a dummy privacy id.
            return self._backend.map(
                col, lambda row: (None, data_extractors.partition_extractor(
                    row), data_extractors.preaggregate_extractor(row)),
                "Extract (partition_key, preaggregate_data))")
        return super()._extract_columns(col, data_extractors)

    def _check_aggregate_params(
        self, col, params: pipeline_dp.AggregateParams,
        data_extractors: Union[pipeline_dp.DataExtractors,
                               pipeline_dp.PreAggregateExtractors]):
        # Do not check data_extractors. The parent implementation does not
        # support PreAggregateExtractors.
        super()._check_aggregate_params(col,
                                        params,
                                        data_extractors=None,
                                        check_data_extractors=False)

    def _annotate(self, col, params, budget):
        # Annotations are not needed because DP computations are not performed.
        return col

    def _drop_partitions_under_threshold(self, col):
        return col


def _check_utility_analysis_params(
    options: analysis.UtilityAnalysisOptions,
    data_extractors: Union[pipeline_dp.DataExtractors,
                           pipeline_dp.PreAggregateExtractors]):
    # Check correctness of data extractors.
    if options.pre_aggregated_data:
        if not isinstance(data_extractors, pipeline_dp.PreAggregateExtractors):
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

    if params.post_aggregation_thresholding:
        raise NotImplementedError("Analysis with post_aggregation_thresholding "
                                  "are not yet implemented")
