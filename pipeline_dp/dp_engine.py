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
import collections
import functools
import numpy as np
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import combiners
from pipeline_dp import contribution_bounders
from pipeline_dp import dp_computations
from pipeline_dp import partition_selection
from pipeline_dp import pipeline_functions
from pipeline_dp import report_generator
from pipeline_dp import sampling_utils
from pipeline_dp.dataset_histograms import computing_histograms
from pipeline_dp.private_contribution_bounds import PrivateL0Calculator

DpNoiseAdditionResult = collections.namedtuple("DpNoiseAdditionResult",
                                               ["noised_value", "noise_stddev"])


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

    def _add_report_generator(self,
                              params,
                              method_name: str,
                              is_public_partition: Optional[bool] = None):
        self._report_generators.append(
            report_generator.ReportGenerator(params, method_name,
                                             is_public_partition))

    def _add_report_stage(self, stage_description):
        self._current_report_generator.add_stage(stage_description)

    def _add_report_stages(self, stages_description):
        for stage_description in stages_description:
            self._add_report_stage(stage_description)

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
        self._check_budget_accountant_compatibility(
            public_partitions is not None, params.metrics,
            params.custom_combiners is not None)

        with self._budget_accountant.scope(weight=params.budget_weight):
            self._add_report_generator(params, "aggregate", public_partitions
                                       is not None)
            if out_explain_computation_report is not None:
                out_explain_computation_report._set_report_generator(
                    self._current_report_generator)
            col = self._aggregate(col, params, data_extractors,
                                  public_partitions)
            budgets = self._budget_accountant._compute_budgets_for_aggregation(
                params.budget_weight)
            return self._annotate(col, params=params, budget=budgets)

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

        col = self._extract_columns(col, data_extractors)
        # col : (privacy_id, partition_key, value)
        if (public_partitions is not None and
                not params.public_partitions_already_filtered):
            col = self._drop_partitions(col,
                                        public_partitions,
                                        partition_extractor=lambda row: row[1])
            self._add_report_stage(
                f"Public partition selection: dropped non public partitions")
        if not params.contribution_bounds_already_enforced:
            contribution_bounder = self._create_contribution_bounder(
                params, combiner.expects_per_partition_sampling())
            col = contribution_bounder.bound_contributions(
                col, params, self._backend, self._current_report_generator,
                combiner.create_accumulator)
            # col : ((privacy_id, partition_key), accumulator)

            col = self._backend.map_tuple(col, lambda pid_pk, v: (pid_pk[1], v),
                                          "Drop privacy id")
            # col : (partition_key, accumulator)
        else:
            col = self._backend.map_tuple(col, lambda pid, pk, v: (pk, v),
                                          "Drop privacy id")
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

        if (public_partitions is None and
                not params.post_aggregation_thresholding and
                not params.public_partitions_already_filtered):
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
                params.partition_selection_strategy, params.pre_threshold)
        # col : (partition_key, accumulator)

        # Compute DP metrics.
        self._add_report_stages(combiner.explain_computation())
        col = self._backend.map_values(col, combiner.compute_metrics,
                                       "Compute DP metrics")

        if params.post_aggregation_thresholding:
            col = self._drop_partitions_under_threshold(col)

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
        self._check_budget_accountant_compatibility(False, [], False)

        with self._budget_accountant.scope(weight=params.budget_weight):
            self._add_report_generator(params, "select_partitions")
            if params.partition_selection_strategy.is_weighted_gaussian:
                return self._select_partitions_weighted_gaussian(
                    col, params, data_extractors)
            col = self._select_partitions(col, params, data_extractors)
            budgets = self._budget_accountant._compute_budgets_for_aggregation(
                params.budget_weight)
            return self._annotate(col, params=params, budget=budgets)

    def _select_partitions(self, col,
                           params: pipeline_dp.SelectPartitionsParams,
                           data_extractors: pipeline_dp.DataExtractors):
        """Implementation of select_partitions computational graph."""
        max_partitions_contributed = params.max_partitions_contributed

        if params.contribution_bounds_already_enforced:
            col = self._backend.map(col, data_extractors.partition_extractor,
                                    "Extract partition_key")
            # col: partition_key
        else:
            # Perform contribution bounding
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
                return sampling_utils.choose_from_list_without_replacement(
                    unique_pks, max_partitions_contributed)

            col = self._backend.flat_map(
                col, sample_unique_elements_fn,
                "Sample cross-partition contributions")
            # col : partition_key

        # A compound accumulator without any child accumulators is used to
        # calculate the raw privacy ID count.
        compound_combiner = combiners.CompoundCombiner([],
                                                       return_named_tuple=False)
        col = self._backend.map(
            col, lambda pk: (pk, compound_combiner.create_accumulator([])),
            "Create accumulator")
        # col : (partition_key, accumulator)

        col = self._backend.combine_accumulators_per_key(
            col, compound_combiner, "Combine accumulators per partition key")
        # col : (partition_key, accumulator)

        col = self._select_private_partitions_internal(
            col,
            max_partitions_contributed,
            max_rows_per_privacy_id=1,
            strategy=params.partition_selection_strategy,
            pre_threshold=params.pre_threshold)
        col = self._backend.keys(col,
                                 "Drop accumulators, keep only partition keys")

        return col

    def _drop_partitions(self, col, partitions, partition_extractor: Callable):
        """Drops partitions in `col` which are not in `partitions`."""
        col = pipeline_functions.key_by(self._backend, col, partition_extractor,
                                        "Key by partition")
        col = self._backend.filter_by_key(col, partitions,
                                          "Filtering out partitions")
        return self._backend.values(col, "Drop key")

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
            strategy: pipeline_dp.PartitionSelectionStrategy,
            pre_threshold: Optional[int]):
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
        budget = self._budget_accountant.request_budget(strategy.mechanism_type)

        def filter_fn(
            budget: 'MechanismSpec', max_partitions: int,
            max_rows_per_privacy_id: int,
            strategy: pipeline_dp.PartitionSelectionStrategy,
            pre_threshold: Optional[int],
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

            selector = _create_partition_selector(strategy, budget,
                                                  max_partitions, pre_threshold)

            return selector.should_keep(privacy_id_count)

        # make filter_fn serializable
        filter_fn = functools.partial(filter_fn, budget,
                                      max_partitions_contributed,
                                      max_rows_per_privacy_id, strategy,
                                      pre_threshold)
        pre_threshold_str = f", pre_threshold={pre_threshold}" if pre_threshold else ""

        def generate_partition_selection_text() -> str:
            if budget.standard_deviation_is_set:
                # PLD case for thresholding.
                noise_stddev = budget.noise_standard_deviation
                thresholding_delta = budget.thresholding_delta
                parameters = f"{noise_stddev=}, {thresholding_delta=}"
            else:
                epsilon = budget.eps
                delta = budget.delta
                parameters = f"{epsilon=}, delta={delta=}"
            text = f"""Private Partition selection: using {strategy.value}
            f"method with ({parameters}, {pre_threshold_str})"""
            return text

        self._add_report_stage(generate_partition_selection_text)

        return self._backend.filter(col, filter_fn, "Filter private partitions")

    def _create_compound_combiner(
            self,
            params: pipeline_dp.AggregateParams) -> combiners.CompoundCombiner:
        """Creates CompoundCombiner based on aggregation parameters."""
        return combiners.create_compound_combiner(params,
                                                  self._budget_accountant)

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams,
        expects_per_partition_sampling: bool
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder based on aggregation parameters."""
        if params.max_contributions:
            return \
                contribution_bounders.SamplingPerPrivacyIdContributionBounder(
                )
        if params.perform_cross_partition_contribution_bounding:
            if expects_per_partition_sampling:
                return contribution_bounders.SamplingCrossAndPerPartitionContributionBounder(
                )
            return contribution_bounders.SamplingCrossPartitionContributionBounder(
            )
        # no cross partition contribution
        if expects_per_partition_sampling:
            return contribution_bounders.LinfSampler()
        # No sampling, but combiners themselves do per partition contribution
        # bounding.
        return contribution_bounders.NoOpSampler()

    def _extract_columns(self, col,
                         data_extractors: pipeline_dp.DataExtractors):
        """Extract columns using data_extractors."""
        if data_extractors.privacy_id_extractor is None:
            # This is the case when contribution bounding already enforced and
            # no need to extract privacy_id.
            privacy_id_extractor = lambda row: None
        else:
            privacy_id_extractor = data_extractors.privacy_id_extractor
        return self._backend.map(
            col, lambda row:
            (privacy_id_extractor(row), data_extractors.partition_extractor(
                row), data_extractors.value_extractor(row)),
            "Extract (privacy_id, partition_key, value))")

    def _check_aggregate_params(self,
                                col,
                                params: pipeline_dp.AggregateParams,
                                data_extractors: pipeline_dp.DataExtractors,
                                check_data_extractors: bool = True):
        if params.max_contributions is not None:
            supported_metrics = [
                pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT,
                pipeline_dp.Metrics.SUM, pipeline_dp.Metrics.MEAN
            ]
            not_supported_metrics = set(
                params.metrics).difference(supported_metrics)
            if not_supported_metrics:
                raise NotImplementedError(
                    f"max_contributions is not supported for {not_supported_metrics}"
                )
        _check_col(col)
        if params is None:
            raise ValueError("params must be set to a valid AggregateParams")
        if not isinstance(params, pipeline_dp.AggregateParams):
            raise TypeError("params must be set to a valid AggregateParams")
        if check_data_extractors:
            _check_data_extractors(data_extractors)
        if params.contribution_bounds_already_enforced:
            if pipeline_dp.Metrics.PRIVACY_ID_COUNT in params.metrics:
                raise ValueError(
                    "PRIVACY_ID_COUNT cannot be computed when "
                    "contribution_bounds_already_enforced is True.")
        if params.post_aggregation_thresholding:
            if pipeline_dp.Metrics.PRIVACY_ID_COUNT not in params.metrics:
                raise ValueError("When post_aggregation_thresholding = True, "
                                 "PRIVACY_ID_COUNT must be in metrics")

    def calculate_private_contribution_bounds(
            self,
            col,
            params: pipeline_dp.CalculatePrivateContributionBoundsParams,
            data_extractors: pipeline_dp.DataExtractors,
            partitions: Any,
            partitions_already_filtered: bool = False):
        """Computes contribution bounds in a differentially private way.

        Computes contribution bounds for COUNT and PRIVACY_ID_COUNT
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
            col = self._drop_partitions(col, partitions,
                                        data_extractors.partition_extractor)

        histograms = computing_histograms.compute_dataset_histograms(
            col, data_extractors, self._backend)
        l0_calculator = PrivateL0Calculator(params, partitions, histograms,
                                            self._backend)
        return pipeline_functions.collect_to_container(
            self._backend,
            {"max_partitions_contributed": l0_calculator.calculate()},
            pipeline_dp.PrivateContributionBounds,
            "Collect calculated private contribution bounds into "
            "PrivateContributionBounds dataclass")

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

    def _check_budget_accountant_compatibility(self, is_public_partition: bool,
                                               metrics: Sequence[
                                                   pipeline_dp.Metric],
                                               custom_combiner: bool) -> None:
        if isinstance(self._budget_accountant,
                      pipeline_dp.NaiveBudgetAccountant):
            # All aggregations support NaiveBudgetAccountant.
            return
        supported_metrics = [
            pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT,
            pipeline_dp.Metrics.SUM, pipeline_dp.Metrics.MEAN
        ]
        non_supported_metrics = set(metrics) - set(supported_metrics)
        if non_supported_metrics:
            raise NotImplementedError(f"Metrics {non_supported_metrics} do not "
                                      f"support PLD budget accounting")
        if custom_combiner:
            raise ValueError(f"PLD budget accounting does not support custom "
                             f"combiners")

    def _drop_partitions_under_threshold(self, col):
        self._add_report_stage("Drop partitions which have noised "
                               "privacy_id_count less than threshold.")
        return self._backend.filter(col,
                                    lambda row: row[1].privacy_id_count != None,
                                    "Drop partitions under threshold")

    def add_dp_noise(self,
                     col,
                     params: pipeline_dp.aggregate_params.AddDPNoiseParams,
                     out_explain_computation_report: Optional[
                         pipeline_dp.ExplainComputationReport] = None):
        """Adds DP noise to the aggregated data.

        This method allows applying differential privacy to pre-aggregated data.
        This relies on the assumption that the sensitivities of the
        pre-aggregated values are known, and the partition keys are public or
        generated with DPEngine.select_partitions.

        Important: unlike the other methods, this method does not enforce the
        sensitivity by contribution bounding and relies on the caller to ensure
        the provided data satisfies the provided bound.

        Args:
          col: collection with elements (partition_key, value). Where value has
            a number type or np.ndarray. It is assumed that all partition_key
            are different.
          params: specifies parameters for noise addition.
          out_explain_computation_report: an output argument, if specified,
            it will contain the Explain Computation report for this aggregation.
            For more details see the docstring to report_generator.py.
        Returns:
            In case of params.output_noise_stddev it returns collection of
            (partition_key, DpNoiseAdditionResult) else
            collection of (partition_key, value + noise).
            Output partition keys are the same as in the input collection.
        """
        # Request budget and create Sensitivities object
        with self._budget_accountant.scope(weight=params.budget_weight):
            self._add_report_generator(params,
                                       "add_dp_noise",
                                       is_public_partition=True)
            if out_explain_computation_report is not None:
                out_explain_computation_report._set_report_generator(
                    self._current_report_generator)
            anonymized_col = self._add_dp_noise(col, params)
            budgets = self._budget_accountant._compute_budgets_for_aggregation(
                params.budget_weight)
            return self._annotate(anonymized_col, params=params, budget=budgets)

    def _add_dp_noise(self, col,
                      params: pipeline_dp.aggregate_params.AddDPNoiseParams):
        mechanism_type = params.noise_kind.convert_to_mechanism_type()
        mechanism_spec = self._budget_accountant.request_budget(mechanism_type)
        sensitivities = dp_computations.Sensitivities(
            l0=params.l0_sensitivity,
            linf=params.linf_sensitivity,
            l1=params.l1_sensitivity,
            l2=params.l2_sensitivity)

        # Add noise to values.
        def create_mechanism() -> dp_computations.AdditiveMechanism:
            return dp_computations.create_additive_mechanism(
                mechanism_spec, sensitivities)

        self._add_report_stage(
            lambda: f"Adding {create_mechanism().noise_kind} noise with "
            f"parameter {create_mechanism().noise_parameter}")

        if params.output_noise_stddev:

            def add_noise(value: Union[int, float]) -> DpNoiseAdditionResult:
                mechanism = create_mechanism()
                return DpNoiseAdditionResult(mechanism.add_noise(value),
                                             mechanism.std)
        else:

            def add_noise(value: Union[int, float]) -> float:
                return create_mechanism().add_noise(value)

        anonymized_col = self._backend.map_values(col, add_noise, "Add noise")

        budget = self._budget_accountant._compute_budgets_for_aggregation(
            params.budget_weight)
        return self._annotate(anonymized_col, params=params, budget=budget)

    def _annotate(self, col, params: Union[pipeline_dp.AggregateParams,
                                           pipeline_dp.SelectPartitionsParams,
                                           pipeline_dp.AddDPNoiseParams],
                  budget: budget_accounting.Budget):
        return self._backend.annotate(col,
                                      "annotation",
                                      params=params,
                                      budget=budget)

    def _weighted_gaussian_calculate_weights(
            self, col, data_extractors: pipeline_dp.DataExtractors,
            max_partitions_contributed: int):
        """Calculate weights for weighted gaussian partition selection.

        Args:
            col: collection of values from which privacy_ids and partition_keys
              can be extracted.
            data_extractors: a pipeline_dp.DataExtractors for the values in col.
            max_partitions_contributed: the maximum number of partitions a
              privacy id can contribute to.
        Returns:
            A collection of (partition_key, weight) pairs.
        """
        col = self._backend.map(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row)),
            "Extract (privacy_id, partition_key))")
        col = self._backend.distinct(col, "Dedup (privacy_id, partition_key)")
        col = self._backend.sample_fixed_per_key(
            col, max_partitions_contributed, "Group pks by pid with sampling")

        def weight(pks):
            return 1.0 / np.sqrt(len(pks))

        col = self._backend.flat_map(
            col, lambda row: [(pk, weight(row[1])) for pk in row[1]],
            "Compute weight per contribution per partition")
        col = self._backend.sum_per_key(col, "Sum weights per partition")
        return col

    def _select_partitions_weighted_gaussian(
            self, col, params: pipeline_dp.SelectPartitionsParams,
            data_extractors: pipeline_dp.DataExtractors):
        """Selects partitions using the weighted gaussian mechanism."""
        col = self._weighted_gaussian_calculate_weights(
            col, data_extractors, params.max_partitions_contributed)
        budget = self._budget_accountant.request_budget(
            params.partition_selection_strategy.mechanism_type)

        def filter_fn(row: Tuple[Any, float]) -> bool:
            partition_selector = (
                partition_selection.create_weighted_gaussian_thresholding(
                    budget.eps, budget.delta,
                    params.max_partitions_contributed))
            return partition_selector.should_keep(row[1])

        col = self._backend.filter(col, filter_fn, "Filter partitions")
        return self._backend.map(col, lambda row: row[0],
                                 "Extract partition keys")


def _check_col(col):
    if col is None or not col:
        raise ValueError("col must be non-empty")


def _check_data_extractors(data_extractors: pipeline_dp.DataExtractors):
    if data_extractors is None:
        raise ValueError("data_extractors must be set to a DataExtractors")
    if not isinstance(data_extractors, pipeline_dp.DataExtractors):
        raise TypeError("data_extractors must be set to a DataExtractors")


def _create_partition_selector(strategy: pipeline_dp.PartitionSelectionStrategy,
                               budget: budget_accounting.MechanismSpec,
                               max_partitions: int, pre_threshold: int):
    if budget.standard_deviation_is_set:
        assert strategy.is_thresholding
        if strategy == pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING:
            return partition_selection.create_laplace_thresholding(
                budget.noise_standard_deviation, budget.thresholding_delta,
                max_partitions, pre_threshold)
        return partition_selection.create_gaussian_thresholding(
            budget.noise_standard_deviation, budget.thresholding_delta,
            max_partitions, pre_threshold)

    return partition_selection.create_partition_selection_strategy(
        strategy, budget.eps, budget.delta, max_partitions, pre_threshold)
