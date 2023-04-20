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
"""DP aggregations."""
import functools
from typing import Any, Optional, Tuple

import pipeline_dp
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
from pipeline_dp import partition_selection
from pipeline_dp import report_generator
from pipeline_dp import sampling_utils
from pipeline_dp.histograms import compute_dataset_histograms
from pipeline_dp import pipeline_functions
from pipeline_dp.private_contribution_bounds import PrivateL0Calculator


class DPEngine:
    """Performs DP aggregations."""

    def __init__(self, budget_accountant: 'BudgetAccountant',
                 backend: 'PipelineBackend'):
        self._budget_accountant = budget_accountant
        self._backend = backend
        self._report_generators = []

    @property
    def _current_report_generator(self):
        return self._report_generators[-1]

    def _add_report_stage(self, stage_description):
        self._current_report_generator.add_stage(stage_description)

    def _add_report_stages(self, stages_description):
        for stage_description in stages_description:
            self._add_report_stage(stage_description)

    def calculate_private_contribution_bounds(
            self,
            col,
            params: pipeline_dp.CalculatePrivateContributionBoundsParams,
            data_extractors: pipeline_dp.DataExtractors,
            partitions: Any,
            partitions_already_filtered: bool = False):
        """Computes contribution bounds for COUNT and PRIVACY_ID_COUNT
        metrics in a differentially private way.
        Currently only max_partitions_contributed is calculated.

        WARNINGS:
          * This API is experimental, there is a possibility that it will
            slightly change in the future.
          * It is supported only for COUNT and PRIVACY_ID_COUNT.
          * It is supported only on Beam and Local backends.

        Args:
          col: collection where all elements are of the same type.
          params: specifies computation parameters necessary for the algorithm.
          data_extractors: functions that extract needed pieces of
            information from elements of 'col'.
          partitions: A collection of partition keys that will be present in
            the result. It can be either the list of public partitions or
            private partitions that were selected before calling this function.
          partitions_already_filtered: if false, then filtering will be made
            and only provided partitions will be kept in col. You can set it to
            true if you have already filtered for these partitions (e.g. you
            did partition selection), it will save you some computation time.

        Returns:
          Collection consisting of 1 element:
          pipeline_dp.PrivateContributionBounds.
        """
        self._check_calculate_private_contribution_bounds_params(
            col, params, data_extractors)

        if not partitions_already_filtered:
            col = self._drop_partitions(col, partitions, data_extractors)

        histograms = compute_dataset_histograms(col, data_extractors,
                                                self._backend)
        l0_calculator = PrivateL0Calculator(params, partitions, histograms,
                                            self._backend)
        return pipeline_composite_functions.collect_to_container(
            self._backend,
            {"max_partitions_contributed": l0_calculator.calculate()},
            pipeline_dp.PrivateContributionBounds,
            "Collect calculated private contribution bounds into "
            "PrivateContributionBounds dataclass")

    def explain_computations_report(self):
        return [
            report_generator.report()
            for report_generator in self._report_generators
        ]

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: pipeline_dp.DataExtractors,
                  public_partitions=None,
                  out_explain_computation_report: Optional[
                      pipeline_dp.ExplainComputationReport] = None):
        """Computes DP aggregate metrics.

        Args:
          col: collection where all elements are of the same type.
          params: specifies which metrics to compute and computation parameters.
          data_extractors: functions that extract needed pieces of information
            from elements of 'col'.
          public_partitions: A collection of partition keys that will be present
            in the result. If not provided, partitions will be selected in a DP
            manner.
          out_explain_computation_report: an output argument, if specified,
            it will contain the Explain Computation report for this aggregation.
            For more details see the docstring to report_generator.py.

        Returns:
          Collection of (partition_key, result_dictionary), where
          'result_dictionary' contains computed metrics per partition_key.
          Keys of 'result_dictionary' correspond to computed metrics, e.g.
          'count' for COUNT metrics etc.
        """
        self._check_aggregate_params(col, params, data_extractors)

        with self._budget_accountant.scope(weight=params.budget_weight):
            self._report_generators.append(
                report_generator.ReportGenerator(params, "aggregate",
                                                 public_partitions is not None))
            if out_explain_computation_report is not None:
                out_explain_computation_report._set_report_generator(
                    self._current_report_generator)
            col = self._aggregate(col, params, data_extractors,
                                  public_partitions)
            budget = self._budget_accountant._compute_budget_for_aggregation(
                params.budget_weight)
            return self._backend.annotate(col,
                                          "annotation",
                                          params=params,
                                          budget=budget)

    def _aggregate(self, col, params: pipeline_dp.AggregateParams,
                   data_extractors: pipeline_dp.DataExtractors,
                   public_partitions):

        if params.custom_combiners:
            # TODO(dvadym): after finishing implementation of custom combiners
            # to figure out whether it makes sense to encapsulate creation of
            # combiners in one function instead of considering 2 cases -
            # standard combiners and custom combiners.
            combiner = combiners.create_compound_combiner_with_custom_combiners(
                params, self._budget_accountant, params.custom_combiners)
        else:
            combiner = self._create_compound_combiner(params)

        if (public_partitions is not None and
                not params.public_partitions_already_filtered):
            col = self._drop_partitions(col, public_partitions, data_extractors)
            self._add_report_stage(
                f"Public partition selection: dropped non public partitions")
        if not params.contribution_bounds_already_enforced:
            col = self._extract_columns(col, data_extractors)
            # col : (privacy_id, partition_key, value)
            contribution_bounder = self._create_contribution_bounder(params)
            col = contribution_bounder.bound_contributions(
                col, params, self._backend, self._current_report_generator,
                combiner.create_accumulator)
            # col : ((privacy_id, partition_key), accumulator)

            col = self._backend.map_tuple(col, lambda pid_pk, v: (pid_pk[1], v),
                                          "Drop privacy id")
            # col : (partition_key, accumulator)
        else:
            # Extract the columns.
            col = self._backend.map(
                col, lambda row: (data_extractors.partition_extractor(row),
                                  data_extractors.value_extractor(row)),
                "Extract (partition_key, value))")
            # col : (partition_key, value)

            col = self._backend.map_values(
                col, lambda value: combiner.create_accumulator([value]),
                "Wrap values into accumulators")
            # col : (partition_key, accumulator)

        if public_partitions:
            col = self._add_empty_public_partitions(col, public_partitions,
                                                    combiner.create_accumulator)
        # col : (partition_key, accumulator)

        col = self._backend.combine_accumulators_per_key(
            col, combiner, "Reduce accumulators per partition key")
        # col : (partition_key, accumulator)

        if public_partitions is None:
            # Perform private partition selection.
            max_rows_per_privacy_id = 1

            if params.contribution_bounds_already_enforced:
                # This regime assumes the input data doesn't have privacy IDs,
                # and therefore we didn't group by them and cannot guarantee one
                # row corresponds to exactly one privacy ID.
                max_rows_per_privacy_id = (
                    params.max_contributions or
                    params.max_contributions_per_partition)

            col = self._select_private_partitions_internal(
                col, params.max_partitions_contributed, max_rows_per_privacy_id,
                params.partition_selection_strategy)
        # col : (partition_key, accumulator)

        # Compute DP metrics.
        self._add_report_stages(combiner.explain_computation())
        col = self._backend.map_values(col, combiner.compute_metrics,
                                       "Compute DP metrics")

        return col

    def _check_select_private_partitions(
            self, col, params: pipeline_dp.SelectPartitionsParams,
            data_extractors: pipeline_dp.DataExtractors):
        """Verifies that arguments for select_partitions are correct."""
        if col is None or not col:
            raise ValueError("col must be non-empty")
        if params is None:
            raise ValueError(
                "params must be set to a valid SelectPrivatePartitionsParams")
        if not isinstance(params, pipeline_dp.SelectPartitionsParams):
            raise TypeError(
                "params must be set to a valid SelectPrivatePartitionsParams")
        if not isinstance(params.max_partitions_contributed,
                          int) or params.max_partitions_contributed <= 0:
            raise ValueError("params.max_partitions_contributed must be set "
                             "(to a positive integer)")
        if data_extractors is None:
            raise ValueError(
                "data_extractors must be set to a pipeline_dp.DataExtractors")
        if not isinstance(data_extractors, pipeline_dp.DataExtractors):
            raise TypeError(
                "data_extractors must be set to a pipeline_dp.DataExtractors")

    def select_partitions(self, col, params: pipeline_dp.SelectPartitionsParams,
                          data_extractors: pipeline_dp.DataExtractors):
        """Retrieves a collection of differentially-private partitions.

        Args:
          col: collection where all elements are of the same type.
          params: parameters, see doc for SelectPrivatePartitionsParams.
          data_extractors: functions that extract needed pieces of information
            from elements of 'col'. Only `privacy_id_extractor` and
            `partition_extractor` are required.
            `value_extractor` is not required.
        """
        self._check_select_private_partitions(col, params, data_extractors)

        with self._budget_accountant.scope(weight=params.budget_weight):
            self._report_generators.append(
                report_generator.ReportGenerator(params, "select_partitions"))
            col = self._select_partitions(col, params, data_extractors)
            budget = self._budget_accountant._compute_budget_for_aggregation(
                params.budget_weight)
            return self._backend.annotate(col,
                                          "annotation",
                                          params=params,
                                          budget=budget)

    def _select_partitions(self, col,
                           params: pipeline_dp.SelectPartitionsParams,
                           data_extractors: pipeline_dp.DataExtractors):
        """Implementation of select_partitions computational graph."""
        max_partitions_contributed = params.max_partitions_contributed

        # Extract the columns.
        col = self._backend.map(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row)),
            "Extract (privacy_id, partition_key))")
        # col : (privacy_id, partition_key)

        # Apply cross-partition contribution bounding
        col = self._backend.group_by_key(col, "Group by privacy_id")

        # col : (privacy_id, [partition_key])

        # Note: This may not be scalable if a single privacy ID contributes
        # to _way_ too many partitions.
        def sample_unique_elements_fn(pid_and_pks):
            pid, pks = pid_and_pks
            unique_pks = list(set(pks))
            sampled_elements = \
                sampling_utils.choose_from_list_without_replacement(
                    unique_pks, max_partitions_contributed)
            return ((pid, pk) for pk in sampled_elements)

        col = self._backend.flat_map(col, sample_unique_elements_fn,
                                     "Sample cross-partition contributions")
        # col : (privacy_id, partition_key)

        # A compound accumulator without any child accumulators is used to
        # calculate the raw privacy ID count.
        compound_combiner = combiners.CompoundCombiner([],
                                                       return_named_tuple=False)
        col = self._backend.map_tuple(
            col, lambda pid, pk: (pk, compound_combiner.create_accumulator([])),
            "Drop privacy id and add accumulator")
        # col : (partition_key, accumulator)

        col = self._backend.combine_accumulators_per_key(
            col, compound_combiner, "Combine accumulators per partition key")
        # col : (partition_key, accumulator)

        col = self._select_private_partitions_internal(
            col,
            max_partitions_contributed,
            max_rows_per_privacy_id=1,
            strategy=params.partition_selection_strategy)
        col = self._backend.keys(col,
                                 "Drop accumulators, keep only partition keys")

        return col

    def _drop_partitions(self, col, partitions,
                         data_extractors: pipeline_dp.DataExtractors):
        """Drops partitions in `col` which are not in `public_partitions`."""
        col = self._backend.map(
            col, lambda row: (data_extractors.partition_extractor(row), row),
            "Extract partition id")
        col = self._backend.filter_by_key(col, partitions,
                                          "Filtering out partitions")
        return self._backend.map_tuple(col, lambda k, v: v, "Drop key")

    def _add_empty_public_partitions(self, col, public_partitions,
                                     aggregator_fn):
        """Adds empty accumulators to all `public_partitions` and returns those
        empty accumulators joined with `col`."""
        self._add_report_stage(
            "Adding empty partitions for public partitions that are missing in "
            "data")
        public_partitions = self._backend.to_collection(
            public_partitions, col, "Public partitions to collection")
        empty_accumulators = self._backend.map(
            public_partitions, lambda partition_key:
            (partition_key, aggregator_fn([])), "Build empty accumulators")

        return self._backend.flatten(
            (col, empty_accumulators),
            "Join public partitions with partitions from data")

    def _select_private_partitions_internal(
            self, col, max_partitions_contributed: int,
            max_rows_per_privacy_id: int,
            strategy: pipeline_dp.PartitionSelectionStrategy):
        """Selects and returns private partitions.

        Args:
            col: collection, with types for each element:
                (partition_key, Accumulator)
            max_partitions_contributed: maximum amount of partitions that one
            privacy unit might contribute.
            strategy: which strategy to use for partition selection.

        Returns:
            collection of elements (partition_key, accumulator).
        """
        budget = self._budget_accountant.request_budget(
            mechanism_type=pipeline_dp.MechanismType.GENERIC)

        def filter_fn(
            budget: 'MechanismSpec', max_partitions: int,
            max_rows_per_privacy_id: int,
            strategy: pipeline_dp.PartitionSelectionStrategy,
            row: Tuple[Any,
                       combiners.CompoundCombiner.AccumulatorType]) -> bool:
            """Lazily creates a partition selection strategy and uses it to
            determine which partitions to keep."""
            row_count, _ = row[1]

            def divide_and_round_up(a, b):
                return (a + b - 1) // b

            # A conservative (lower) estimate of how many privacy IDs
            # contributed to this partition. This estimate is only needed when
            # privacy IDs are not available in the original dataset.
            privacy_id_count = divide_and_round_up(row_count,
                                                   max_rows_per_privacy_id)

            strategy_object = \
                partition_selection.create_partition_selection_strategy(
                    strategy, budget.eps, budget.delta, max_partitions)
            return strategy_object.should_keep(privacy_id_count)

        # make filter_fn serializable
        filter_fn = functools.partial(filter_fn, budget,
                                      max_partitions_contributed,
                                      max_rows_per_privacy_id, strategy)
        self._add_report_stage(
            lambda: f"Private Partition selection: using {strategy.value} "
            f"method with (eps={budget.eps}, delta={budget.delta})")

        return self._backend.filter(col, filter_fn, "Filter private partitions")

    def _create_compound_combiner(
            self,
            params: pipeline_dp.AggregateParams) -> combiners.CompoundCombiner:
        """Creates CompoundCombiner based on aggregation parameters."""
        return combiners.create_compound_combiner(params,
                                                  self._budget_accountant)

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder based on aggregation parameters."""
        if params.max_contributions:
            return \
                contribution_bounders.SamplingPerPrivacyIdContributionBounder(
                )
        else:
            return \
                contribution_bounders.SamplingCrossAndPerPartitionContributionBounder(
                )

    def _extract_columns(self, col,
                         data_extractors: pipeline_dp.DataExtractors):
        """Extract columns using data_extractors."""
        return self._backend.map(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row),
                              data_extractors.value_extractor(row)),
            "Extract (privacy_id, partition_key, value))")

    def _check_calculate_private_contribution_bounds_params(
            self,
            col,
            params: pipeline_dp.CalculatePrivateContributionBoundsParams,
            data_extractors: pipeline_dp.DataExtractors,
            check_data_extractors: bool = True):
        _check_col(col)
        if params is None:
            raise ValueError("params must be set to a valid "
                             "CalculatePrivateContributionBoundsParams")
        if not isinstance(params,
                          pipeline_dp.CalculatePrivateContributionBoundsParams):
            raise TypeError("params must be set to a valid "
                            "CalculatePrivateContributionBoundsParams")
        if check_data_extractors:
            _check_data_extractors(data_extractors)

    def _check_aggregate_params(self,
                                col,
                                params: pipeline_dp.AggregateParams,
                                data_extractors: pipeline_dp.DataExtractors,
                                check_data_extractors: bool = True):
        if params.max_contributions is not None:
            raise NotImplementedError("max_contributions is not supported yet.")
        _check_col(col)
        if params is None:
            raise ValueError("params must be set to a valid AggregateParams")
        if not isinstance(params, pipeline_dp.AggregateParams):
            raise TypeError("params must be set to a valid AggregateParams")
        if check_data_extractors:
            _check_data_extractors(data_extractors)
        if params.contribution_bounds_already_enforced:
            if data_extractors.privacy_id_extractor:
                raise ValueError(
                    "privacy_id_extractor should be set iff "
                    "contribution_bounds_already_enforced is False")
            if pipeline_dp.Metrics.PRIVACY_ID_COUNT in params.metrics:
                raise ValueError(
                    "PRIVACY_ID_COUNT cannot be computed when "
                    "contribution_bounds_already_enforced is True.")


def _check_col(col):
    if col is None or not col:
        raise ValueError("col must be non-empty")


def _check_data_extractors(data_extractors: pipeline_dp.DataExtractors):
    if data_extractors is None:
        raise ValueError("data_extractors must be set to a DataExtractors")
    if not isinstance(data_extractors, pipeline_dp.DataExtractors):
        raise TypeError("data_extractors must be set to a DataExtractors")
