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
from typing import Iterable, Optional, Sequence

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
from pipeline_dp import pipeline_backend
import utility_analysis_new.contribution_bounders as utility_contribution_bounders
import utility_analysis_new.combiners as utility_analysis_combiners


@dataclasses.dataclass
class MultiParameterConfiguration:
    """Specifies parameters for multi-parameter Utility Analysis.

    All MultiParameterConfiguration attributes corresponds to attributes in
    pipeline_dp.AggregateParams.

    UtilityAnalysisEngine can perform utility analysis for multiple sets of
    parameters simultaneously. API for this is the following:
    1. Specify blue-print AggregateParams instance.
    2. Set MultiParameterConfiguration attributes (see the example below). Note
    that each attribute is a sequence of parameter values for which the utility
    analysis will be run. All attributes that have non-None values must have
    the same length.
    3. Pass the created objects to UtilityAnalysisEngine.aggregate().

    Example:
        max_partitions_contributed = [1, 2]
        max_contributions_per_partition = [10, 11]

        Then the utility analysis will be performed for
          AggregateParams(max_partitions_contributed=1, max_contributions_per_partition=10)
          AggregateParams(max_partitions_contributed=2, max_contributions_per_partition=11)
    """
    max_partitions_contributed: Sequence[int] = None
    max_contributions_per_partition: Sequence[int] = None
    min_sum_per_partition: Sequence[float] = None
    max_sum_per_partition: Sequence[float] = None

    def __post_init__(self):
        attributes = dataclasses.asdict(self)
        sizes = [len(value) for value in attributes.values() if value]
        if not sizes:
            raise ValueError("MultiParameterConfiguration must have at least 1"
                             " non-empty attribute.")
        if min(sizes) != max(sizes):
            raise ValueError(
                "All set attributes in MultiParameterConfiguration must have "
                "the same length.")
        if (self.min_sum_per_partition is None) != (self.max_sum_per_partition
                                                    is None):
            raise ValueError(
                "MultiParameterConfiguration: min_sum_per_partition and "
                "max_sum_per_partition must be both set or both None.")
        self._size = sizes[0]

    @property
    def size(self):
        return self._size

    def get_aggregate_params(self, params: pipeline_dp.AggregateParams,
                             index: int) -> pipeline_dp.AggregateParams:
        """Returns AggregateParams with the index-th parameters."""
        params = copy.copy(params)
        if self.max_partitions_contributed:
            params.max_partitions_contributed = self.max_partitions_contributed[
                index]
        if self.max_contributions_per_partition:
            params.max_contributions_per_partition = self.max_contributions_per_partition[
                index]
        if self.min_sum_per_partition:
            params.min_sum_per_partition = self.min_sum_per_partition[index]
        if self.max_sum_per_partition:
            params.max_sum_per_partition = self.max_sum_per_partition[index]
        return params


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations."""

    def __init__(self, budget_accountant: budget_accounting.BudgetAccountant,
                 backend: pipeline_backend.PipelineBackend):
        super().__init__(budget_accountant, backend)
        self._is_public_partitions = None

    def aggregate(
        self,
        col,
        params: pipeline_dp.AggregateParams,
        data_extractors: pipeline_dp.DataExtractors,
        public_partitions=None,
        multi_param_configuration: Optional[MultiParameterConfiguration] = None
    ):
        """Performs utility analysis for DP aggregations per partition.

        Args:
          col: collection where all elements are of the same type.
          params: specifies which metrics to compute and computation parameters.
          data_extractors: functions that extract needed pieces of information
            from elements of 'col'.
          public_partitions: A collection of partition keys that will be present
            in the result. If not provided, the utility analysis with private
            partition selection will be performed.
          multi_param_configuration: if provided the utility analysis for
            multiple parameters will be performed: 'params' is used as
            blue-print and non-None attributes from 'multi_param_configuration'
            are used for creating multiple AggregateParams. See docstring for
            MultiParameterConfiguration for more details.

        Returns:
            A collection with elements (pk, utility analysis metrics).
        """
        _check_utility_analysis_params(params, public_partitions)
        self._is_public_partitions = public_partitions is not None
        self._multi_run_configuration = multi_param_configuration
        result = super().aggregate(col, params, data_extractors,
                                   public_partitions)
        self._is_public_partitions = None
        self._multi_run_configuration = None
        return result

    def select_partitions(
        self,
        col,
        params: pipeline_dp.SelectPartitionsParams,
        data_extractors: pipeline_dp.DataExtractors,
        multi_param_configuration: Optional[MultiParameterConfiguration] = None
    ):
        # self._multi_run_configuration = multi_param_configuration
        # self._is_public_partitions = False
        # result = super().select_partitions(col, params, data_extractors)
        # self._is_public_partitions = None
        # self._multi_run_configuration = None
        # return result
        aggregate_params = pipeline_dp.AggregateParams(
            max_partitions_contributed=params.max_partitions_contributed,
            partition_selection_strategy=params.partition_selection_strategy,
            max_contributions_per_partition=1,
            metrics=[])
        return self.aggregate(
            col,
            aggregate_params,
            data_extractors,
            multi_param_configuration=multi_param_configuration)

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
        if self._multi_run_configuration is None:
            yield params
        else:
            for i in range(self._multi_run_configuration.size):
                yield self._multi_run_configuration.get_aggregate_params(
                    params, i)

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
