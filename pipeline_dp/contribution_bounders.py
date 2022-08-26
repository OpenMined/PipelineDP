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
"""Implementation of contribution bounding."""

import abc
import collections
from typing import Callable, Iterable

import pipeline_dp
from pipeline_dp import pipeline_backend
from pipeline_dp import sampling_utils


class ContributionBounder(abc.ABC):
    """Interface for objects which perform contribution bounding."""

    @abc.abstractmethod
    def bound_contributions(self, col, params: pipeline_dp.AggregateParams,
                            backend: pipeline_backend.PipelineBackend,
                            aggregate_fn: Callable):
        """Bound contributions of privacy id.

        Contribution bounding is performed to ensure that sensitivity of the
        aggregations to anonymize are limited. There are many ways how to
        perform contribution bounding. Sub-classes implement specific contribution
        strategies.

        This function also performs the aggregation step per
        (privacy_id, partition_key) by calling aggregate_fn on data points
        corresponding to each (privacy_id, partition_key).

        Args:
          col: collection, with types of each element: (privacy_id,
            partition_key, value).
          params: contains parameters needed for contribution bounding.
          backend: pipeline backend for performing operations on collections.
          aggregate_fn: function that takes a list of values and returns an
            aggregator object.
        Returns:
          collection with elements ((privacy_id, partition_key),
              accumulator).
        """


class SamplingCrossAndPerPartitionContributionBounder(ContributionBounder):
    """Bounds the contribution of privacy_id per and cross partitions.

    It ensures that each privacy_id contributes to not more than
    max_partitions_contributed partitions (cross-partition contribution
    bounding) and not more than max_contributions_per_partition to each
    contributed partition (per-partition contribution bounding), by performing
    sampling if needed.
    """

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """See docstrings for this class and the base class."""
        max_partitions_contributed = params.max_partitions_contributed
        max_contributions_per_partition = params.max_contributions_per_partition
        col = backend.map_tuple(
            col, lambda pid, pk, v: ((pid, pk), v),
            "Rekey to ( (privacy_id, partition_key), value))")
        col = backend.sample_fixed_per_key(
            col, params.max_contributions_per_partition,
            "Sample per (privacy_id, partition_key)")
        report_generator.add_stage(
            f"Per-partition contribution bounding: for each privacy_id and each"
            f"partition, randomly select max(actual_contributions_per_partition"
            f", {max_contributions_per_partition}) contributions.")
        # ((privacy_id, partition_key), [value])
        col = backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn after per partition bounding")
        # ((privacy_id, partition_key), accumulator)
        # Cross partition bounding
        col = backend.map_tuple(
            col, lambda pid_pk, v: (pid_pk[0], (pid_pk[1], v)),
            "Rekey to (privacy_id, (partition_key, accumulator))")
        col = backend.sample_fixed_per_key(col,
                                           params.max_partitions_contributed,
                                           "Sample per privacy_id")

        report_generator.add_stage(
            f"Cross-partition contribution bounding: for each privacy_id "
            f"randomly select max(actual_partition_contributed, "
            f"{max_partitions_contributed}) partitions")

        # (privacy_id, [(partition_key, accumulator)])
        def rekey_by_privacy_id_and_unnest(pid_pk_v):
            pid, pk_values = pid_pk_v
            return (((pid, pk), v) for (pk, v) in pk_values)

        return backend.flat_map(col, rekey_by_privacy_id_and_unnest,
                                "Rekey by privacy_id and unnest")


class SamplingPerPrivacyIdContributionBounder(ContributionBounder):
    """Bounds the total contributions of a privacy_id.

    If a privacy_id contributes more than max_contributions, then
    max_contributions contributions are uniformly sampled, otherwise all
    contributions are kept.
    """

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """See docstrings for this class and the base class."""
        max_contributions = params.max_contributions
        col = backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to ((privacy_id), (partition_key, value))")
        col = backend.sample_fixed_per_key(col, max_contributions,
                                           "Sample per privacy_id")
        report_generator.add_stage(
            f"User contribution bounding: randomly selected not "
            f"more than {max_contributions} contributions")

        # (privacy_id, [(partition_key, value)])

        col = collect_values_per_partition_key_per_privacy_id(col, backend)

        # (privacy_id, [(partition_key, [value])])

        # Rekey it into values per privacy id and partition key.
        # Unnest the list per privacy id.
        def rekey_per_privacy_id_per_partition_key(pid_pk_v_values):
            privacy_id, partition_values = pid_pk_v_values

            for partition_key, values in partition_values:
                yield (privacy_id, partition_key), values

        # Unnest the list per privacy id.
        col = backend.flat_map(col, rekey_per_privacy_id_per_partition_key,
                               "Unnest")
        # ((privacy_id, partition_key), [value])

        return backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn after per privacy_id contribution bounding")


class SamplingCrossPartitionContributionBounder(ContributionBounder):
    """Bounds the contribution of privacy_id cross partitions.

    It ensures that each privacy_id contributes to not more than
    max_partitions_contributed partitions (cross-partition contribution
    bounding), by performing sampling if needed. It is assumed that the provided
    aggregate_fn function bounds per-partition contributions.
    """

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        col = backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to ((privacy_id), (partition_key, value))")

        col = backend.group_by_key(col, "Group by privacy_id")
        # (privacy_id, [(partition_key, value)])

        col = collect_values_per_partition_key_per_privacy_id(col, backend)
        # (privacy_id, [(partition_key, [value])])

        # Bound cross partition contributions with sampling.
        sample = sampling_utils.choose_from_list_without_replacement
        sample_size = params.max_partitions_contributed
        col = backend.map_values(col, lambda a: sample(a, sample_size))

        # (privacy_id, [partition_key, [value]])

        # Unnest the list per privacy id.
        def rekey_per_privacy_id_per_partition_key(pid_pk_v_values):
            privacy_id, partition_values = pid_pk_v_values

            for partition_key, values in partition_values:
                yield (privacy_id, partition_key), values

        col = backend.flat_map(col, rekey_per_privacy_id_per_partition_key,
                               "Unnest per privacy_id")
        # ((privacy_id, partition_key), [value])

        return backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn after cross-partition contribution bounding")


def collect_values_per_partition_key_per_privacy_id(
        col, backend: pipeline_backend.PipelineBackend):
    """Collects values into a list for each privacy_id and partition_key.

    The output collection is a mapping from privacy_id (i.e. each privacy_id
    from 'col' occurs exactly once) to a list [(partition_key, [values]].
    For any privacy_id, each partition_key it contributes to appear exactly
    once.

    Args:
        col: collection with elements (privacy_id,
        Iterable[(partition_key, value)]). It's assumed that each privacy_id
        occurs only once.
        backend: pipeline backend for performing operations on collections.

    Returns:
        collection with elements (privacy_id, [partition_key, [values]]).
    """

    def collect_values_per_partition_key_per_privacy_id_fn(input: Iterable):
        d = collections.defaultdict(list)
        for key, value in input:
            d[key].append(value)
        return list(d.items())

    return backend.map_values(
        col, collect_values_per_partition_key_per_privacy_id_fn,
        "Collect values per privacy_id and partition_key")
