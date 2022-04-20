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

    def aggregate(self, col, params: pipeline_dp.AggregateParams,
                  data_extractors: DataExtractors):
        """Computes DP aggregate metrics.

        Args:
          col: collection where all elements are of the same type.
          params: specifies which metrics to compute and computation parameters.
          data_extractors: functions that extract needed pieces of information
            from elements of 'col'.
        """
        _check_aggregate_params(col, params, data_extractors)

        with self._budget_accountant.scope(weight=params.budget_weight):
            return self._aggregate(col, params, data_extractors)

    def _aggregate(self, col, params: pipeline_dp.AggregateParams,
                   data_extractors: DataExtractors):

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

        if params.public_partitions is not None:
            col = self._drop_not_public_partitions(col,
                                                   params.public_partitions,
                                                   data_extractors)

        # Extract the columns.
        col = self._backend.map(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row),
                              data_extractors.value_extractor(row)),
            "Extract (privacy_id, partition_key, value))")
        # col : (privacy_id, partition_key, value)
        col = self._bound_contributions(col, params.max_partitions_contributed,
                                        params.max_contributions_per_partition,
                                        combiner.create_accumulator)
        # col : ((privacy_id, partition_key), accumulator)

        col = self._backend.map_tuple(col, lambda pid_pk, v: (pid_pk[1], v),
                                      "Drop privacy id")
        # col : (partition_key, accumulator)

        if params.public_partitions:
            col = self._add_empty_public_partitions(col,
                                                    params.public_partitions,
                                                    combiner.create_accumulator)
        # col : (partition_key, accumulator)

        col = self._backend.combine_accumulators_per_key(
            col, combiner, "Reduce accumulators per partition key")
        # col : (partition_key, accumulator)

        if params.public_partitions is None:
            col = self._select_private_partitions_internal(
                col, params.max_partitions_contributed)
        # col : (partition_key, accumulator)

        # Compute DP metrics.
        col = self._backend.map_values(col, combiner.compute_metrics,
                                       "Compute DP` metrics")

        self._backend.annotate()

        return col

    def _check_select_private_partitions(
            self, col, params: pipeline_dp.SelectPartitionsParams,
            data_extractors: DataExtractors):
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
            return self._select_partitions(col, params, data_extractors)

    def _select_partitions(self, col,
                           params: pipeline_dp.SelectPartitionsParams,
                           data_extractors: DataExtractors):
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

        # A compound accumulator without any child accumulators is used to calculate the raw privacy ID count.
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
            col, max_partitions_contributed)
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
            f"Cross-partition contribution bounding: randomly selected not more than "
            f"{max_partitions_contributed} partitions per user")

        # (privacy_id, [(partition_key, accumulator)])
        def unnest_cross_partition_bound_sampled_per_key(pid_pk_v):
            pid, pk_values = pid_pk_v
            return (((pid, pk), v) for (pk, v) in pk_values)

        return self._backend.flat_map(
            col, unnest_cross_partition_bound_sampled_per_key, "Unnest")

    def _select_private_partitions_internal(self, col,
                                            max_partitions_contributed: int):
        """Selects and publishes private partitions.

        Args:
            col: collection, with types for each element:
                (partition_key, Accumulator)
            max_partitions_contributed: maximum amount of partitions that one privacy unit
                might contribute.

        Returns:
            collection of elements (partition_key, accumulator)
        """
        budget = self._budget_accountant.request_budget(
            mechanism_type=pipeline_dp.MechanismType.GENERIC)

        def filter_fn(
            budget: 'MechanismSpec', max_partitions: int,
            row: Tuple[Any,
                       combiners.CompoundCombiner.AccumulatorType]) -> bool:
            """Lazily creates a partition selection strategy and uses it to determine which
            partitions to keep."""
            privacy_id_count, _ = row[1]
            partition_selection_strategy = (
                partition_selection.
                create_truncated_geometric_partition_strategy(
                    budget.eps, budget.delta, max_partitions))
            return partition_selection_strategy.should_keep(privacy_id_count)

        # make filter_fn serializable
        filter_fn = functools.partial(filter_fn, budget,
                                      max_partitions_contributed)
        self._add_report_stage(
            lambda:
            f"Private Partition selection: using {budget.mechanism_type.value} "
            f"method with (eps= {budget.eps}, delta = {budget.delta})")

        return self._backend.filter(col, filter_fn, "Filter private partitions")


def _check_aggregate_params(col, params: pipeline_dp.AggregateParams,
                            data_extractors: DataExtractors):
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
