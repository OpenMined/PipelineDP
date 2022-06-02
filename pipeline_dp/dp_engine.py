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
import dataclasses
import functools
from typing import Any, Callable, Tuple
import numpy as np

import pipeline_dp
from pipeline_dp import combiners
import pipeline_dp.report_generator as report_generator

import pydp.algorithms.partition_selection as partition_selection


@dataclasses.dataclass
class DataExtractors:
    """Data extractors.

    A set of functions that, given a piece of input, return the privacy id, 
    partition key, and value respectively.
    """

    privacy_id_extractor: Callable = None
    partition_extractor: Callable = None
    value_extractor: Callable = None


class DPEngine:
    """Performs DP aggregations."""

    def __init__(self, budget_accountant: 'BudgetAccountant',
                 backend: 'PipelineBackend'):
        self._budget_accountant = budget_accountant
        self._backend = backend
        self._report_generators = []

    def _add_report_stage(self, text):
        self._report_generators[-1].add_stage(text)

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: DataExtractors,
                  public_partitions=None):
        """Computes DP aggregate metrics.

        Args:
          col: collection where all elements are of the same type.
          params: specifies which metrics to compute and computation parameters.
          data_extractors: functions that extract needed pieces of information
          from elements of 'col'.
          public_partitions: A collection of partition keys that will be present
          in the result. If not provided, partitions will be selected in a DP
          manner.

        Returns:
          Collection of (partition_key, result_dictionary), where
          'result_dictionary' contains computed metrics per partition_key.
          Keys of 'result_dictionary' correspond to computed metrics, e.g.
          'count' for COUNT metrics etc.
        """
        _check_aggregate_params(col, params, data_extractors)

        with self._budget_accountant.scope(weight=params.budget_weight):
            col = self._aggregate(col, params, data_extractors,
                                  public_partitions)
            budget = self._budget_accountant._compute_budget_for_aggregation(
                params.budget_weight)
            return self._backend.annotate(col,
                                          "annotation",
                                          params=params,
                                          budget=budget)

    def _aggregate(self, col, params: pipeline_dp.AggregateParams,
                   data_extractors: DataExtractors, public_partitions):

        self._report_generators.append(report_generator.ReportGenerator(params))

        if params.custom_combiners:
            # TODO(dvadym): after finishing implementation of custom combiners
            # to figure out whether it makes sense to encapsulate creation of
            # combiners in one function instead of considering 2 cases -
            # standard combiners and custom combiners.
            combiner = combiners.create_compound_combiner_with_custom_combiners(
                params, self._budget_accountant, params.custom_combiners)
        else:
            combiner = combiners.create_compound_combiner(
                params, self._budget_accountant)

        if public_partitions is not None:
            col = self._drop_not_public_partitions(col, public_partitions,
                                                   data_extractors)
        if not params.contribution_bounds_already_enforced:
            # Extract the columns.
            col = self._backend.map(
                col, lambda row: (data_extractors.privacy_id_extractor(row),
                                  data_extractors.partition_extractor(row),
                                  data_extractors.value_extractor(row)),
                "Extract (privacy_id, partition_key, value))")
            # col : (privacy_id, partition_key, value)
            if params.max_contributions:
                col = self._bound_per_privacy_id_contributions(
                    col, params.max_contributions, combiner.create_accumulator)
            else:
                col = self._bound_contributions(
                    col, params.max_partitions_contributed,
                    params.max_contributions_per_partition,
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
            max_rows_per_privacy_id = 1

            if params.contribution_bounds_already_enforced:
                # This regime assumes the input data doesn't have privacy IDs,
                # and therefore we didn't group by them and cannot guarantee one
                # row corresponds to exactly one privacy ID.
                max_rows_per_privacy_id = params.max_contributions_per_partition

            col = self._select_private_partitions_internal(
                col, params.max_partitions_contributed, max_rows_per_privacy_id)
        # col : (partition_key, accumulator)

        # Compute DP metrics.
        col = self._backend.map_values(col, combiner.compute_metrics,
                                       "Compute DP` metrics")

        return col

    def _check_select_private_partitions(
            self, col, params: pipeline_dp.SelectPartitionsParams,
            data_extractors: DataExtractors):
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
            raise ValueError("data_extractors must be set to a DataExtractors")
        if not isinstance(data_extractors, pipeline_dp.DataExtractors):
            raise TypeError("data_extractors must be set to a DataExtractors")

    def select_partitions(self, col, params: pipeline_dp.SelectPartitionsParams,
                          data_extractors: DataExtractors):
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
            col = self._select_partitions(col, params, data_extractors)
            budget = self._budget_accountant._compute_budget_for_aggregation(
                params.budget_weight)
            return self._backend.annotate(col,
                                          "annotation",
                                          params=params,
                                          budget=budget)

    def _select_partitions(self, col,
                           params: pipeline_dp.SelectPartitionsParams,
                           data_extractors: DataExtractors):
        """Implementation of select_partitions computational graph."""
        self._report_generators.append(report_generator.ReportGenerator(params))
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
            if len(unique_pks) <= max_partitions_contributed:
                sampled_elements = unique_pks
            else:
                # np.random.choice makes casting of elements to numpy types
                # which is undesirable by 2 reasons:
                # 1. Apache Beam can not serialize numpy types.
                # 2. It might lead for losing precision (e.g. arbitrary
                # precision int is converted to int64).
                # So np.random.choice should not be applied directly to
                # 'unique_pks'. It is better to apply it to indices.
                sampled_indices = np.random.choice(np.arange(len(unique_pks)),
                                                   max_partitions_contributed,
                                                   replace=False)

                sampled_elements = [unique_pks[i] for i in sampled_indices]

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
            col, max_partitions_contributed, max_rows_per_privacy_id=1)
        col = self._backend.keys(col,
                                 "Drop accumulators, keep only partition keys")

        return col

    def _drop_not_public_partitions(self, col, public_partitions,
                                    data_extractors: DataExtractors):
        """Drops partitions in `col` which are not in `public_partitions`."""
        col = self._backend.map(
            col, lambda row: (data_extractors.partition_extractor(row), row),
            "Extract partition id")
        col = self._backend.filter_by_key(
            col, public_partitions, "Filtering out non-public partitions")
        self._add_report_stage(
            f"Public partition selection: dropped non public partitions")
        return self._backend.map_tuple(col, lambda k, v: v, "Drop key")

    def _add_empty_public_partitions(self, col, public_partitions,
                                     aggregator_fn):
        """Adds empty accumulators to all `public_partitions` and returns those
        empty accumulators joined with `col`."""
        self._add_report_stage(
            "Adding empty partitions to public partitions that are missing in "
            "data")
        empty_accumulators = self._backend.map(
            public_partitions, lambda partition_key:
            (partition_key, aggregator_fn([])), "Build empty accumulators")

        return self._backend.flatten(
            col, empty_accumulators,
            "Join public partitions with partitions from data")

    def _bound_per_privacy_id_contributions(self, col, max_contributions: int,
                                            aggregator_fn):
        """Bounds the total contributions by a privacy_id.

        Args:
          col: collection, with types of each element: (privacy_id,
            partition_key, value).
          max_contributions: maximum number of records that one privacy id can
            contribute.
          aggregator_fn: function that takes a list of values and returns an
            aggregator object which handles all aggregation logic.

        return: collection with elements ((privacy_id, partition_key),
              accumulator).
        """
        col = self._backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to ((privacy_id), (partition_key, value))")
        col = self._backend.sample_fixed_per_key(col, max_contributions,
                                                 "Sample per privacy_id")
        self._add_report_stage(
            f"User contributions bounding: randomly selected not "
            f"more than {max_contributions} contributions")

        # (privacy_id, [(partition_key, value)])

        # Convert the per privacy id list into a dict with key as partition_key
        # and value to be list of input values.
        def collect_values_per_partition_key_per_privacy_id(input_list):
            d = collections.defaultdict(list)
            for key, value in input_list:
                d[key].append(value)
            return d

        col = self._backend.map_values(
            col, collect_values_per_partition_key_per_privacy_id,
            "Group per (privacy_id, partition_key)")

        # (privacy_id, {partition_key: [value]})

        # Rekey it into values per privacy id and partition key.
        def rekey_per_privacy_id_per_partition_key(pid_pk_dict):
            privacy_id, partition_dict = pid_pk_dict
            for partition_key, values in partition_dict.items():
                yield (privacy_id, partition_key), values

        # Unnest the list per privacy id.
        col = self._backend.flat_map(col,
                                     rekey_per_privacy_id_per_partition_key,
                                     "Unnest")
        # ((privacy_id, partition_key), [value])

        return self._backend.map_values(
            col, aggregator_fn,
            "Apply aggregate_fn after per privacy id contributions bounding")

    def _bound_contributions(self, col, max_partitions_contributed: int,
                             max_contributions_per_partition: int,
                             aggregator_fn):
        """Bounds the contribution by privacy_id in and cross partitions.

        Args:
          col: collection, with types of each element: (privacy_id,
            partition_key, value).
          max_partitions_contributed: maximum number of partitions that one
            privacy id can contribute to.
          max_contributions_per_partition: maximum number of records that one
            privacy id can contribute to one partition.
          aggregator_fn: function that takes a list of values and returns an
            aggregator object which handles all aggregation logic.

        return: collection with elements ((privacy_id, partition_key),
              accumulator).
        """
        # per partition-contribution bounding with bounding of each contribution
        col = self._backend.map_tuple(
            col, lambda pid, pk, v: ((pid, pk), v),
            "Rekey to ( (privacy_id, partition_key), value))")
        col = self._backend.sample_fixed_per_key(
            col, max_contributions_per_partition,
            "Sample per (privacy_id, partition_key)")
        self._add_report_stage(
            f"Per-partition contribution bounding: randomly selected not "
            f"more than {max_contributions_per_partition} contributions")
        # ((privacy_id, partition_key), [value])
        col = self._backend.map_values(
            col, aggregator_fn,
            "Apply aggregate_fn after per partition bounding")
        # ((privacy_id, partition_key), accumulator)
        # Cross partition bounding
        col = self._backend.map_tuple(
            col, lambda pid_pk, v: (pid_pk[0], (pid_pk[1], v)),
            "Rekey to (privacy_id, (partition_key, accumulator))")
        col = self._backend.sample_fixed_per_key(col,
                                                 max_partitions_contributed,
                                                 "Sample per privacy_id")

        self._add_report_stage(
            f"Cross-partition contribution bounding: randomly selected not more"
            f" than {max_partitions_contributed} partitions per privacy id")

        # (privacy_id, [(partition_key, accumulator)])
        def unnest_cross_partition_bound_sampled_per_key(pid_pk_v):
            pid, pk_values = pid_pk_v
            return (((pid, pk), v) for (pk, v) in pk_values)

        return self._backend.flat_map(
            col, unnest_cross_partition_bound_sampled_per_key, "Unnest")

    def _select_private_partitions_internal(self, col,
                                            max_partitions_contributed: int,
                                            max_rows_per_privacy_id: int):
        """Selects and publishes private partitions.

        Args:
            col: collection, with types for each element:
                (partition_key, Accumulator)
            max_partitions_contributed: maximum amount of partitions that one
            privacy unit might contribute.

        Returns:
            collection of elements (partition_key, accumulator).
        """
        budget = self._budget_accountant.request_budget(
            mechanism_type=pipeline_dp.MechanismType.GENERIC)

        def filter_fn(
            budget: 'MechanismSpec', max_partitions: int,
            max_rows_per_privacy_id: int,
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

            partition_selection_strategy = (
                partition_selection.
                create_truncated_geometric_partition_strategy(
                    budget.eps, budget.delta, max_partitions))
            return partition_selection_strategy.should_keep(privacy_id_count)

        # make filter_fn serializable
        filter_fn = functools.partial(filter_fn, budget,
                                      max_partitions_contributed,
                                      max_rows_per_privacy_id)
        self._add_report_stage(
            lambda:
            f"Private Partition selection: using {budget.mechanism_type.value} "
            f"method with (eps= {budget.eps}, delta = {budget.delta})")

        return self._backend.filter(col, filter_fn, "Filter private partitions")


def _check_aggregate_params(col, params: pipeline_dp.AggregateParams,
                            data_extractors: DataExtractors):
    if params.max_contributions is not None:
        raise NotImplementedError("max_contributions is not supported yet.")
    if col is None or not col:
        raise ValueError("col must be non-empty")
    if params is None:
        raise ValueError("params must be set to a valid AggregateParams")
    if not isinstance(params, pipeline_dp.AggregateParams):
        raise TypeError("params must be set to a valid AggregateParams")
    if data_extractors is None:
        raise ValueError("data_extractors must be set to a DataExtractors")
    if not isinstance(data_extractors, pipeline_dp.DataExtractors):
        raise TypeError("data_extractors must be set to a DataExtractors")
    if params.contribution_bounds_already_enforced == \
            (data_extractors.privacy_id_extractor is not None):
        raise ValueError("privacy_id_extractor should be set iff "\
                         "contribution_bounds_already_enforced is False")
