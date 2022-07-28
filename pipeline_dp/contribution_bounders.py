import abc
import collections
from typing import Callable

import pipeline_dp
from pipeline_dp import pipeline_backend


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
    """Bounds the contribution by privacy_id per and cross partitions.

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
            f"Cross-partition contribution bounding: for each privacy id "
            f"randomly select max(actual_partition_contributed, "
            f"{max_partitions_contributed}) partitions")

        # (privacy_id, [(partition_key, accumulator)])
        def rekey_by_privacy_id_and_unnest(pid_pk_v):
            pid, pk_values = pid_pk_v
            return (((pid, pk), v) for (pk, v) in pk_values)

        return backend.flat_map(col, rekey_by_privacy_id_and_unnest,
                                "Rekey by privacy id and unnest")


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

        # Convert the per privacy id list into a dict with key as partition_key
        # and values as the list of input values.
        def collect_values_per_partition_key_per_privacy_id(input_list):
            d = collections.defaultdict(list)
            for key, value in input_list:
                d[key].append(value)
            return d

        col = backend.map_values(
            col, collect_values_per_partition_key_per_privacy_id,
            "Group per (privacy_id, partition_key)")

        # (privacy_id, {partition_key: [value]})

        # Rekey it into values per privacy id and partition key.
        def rekey_per_privacy_id_per_partition_key(pid_pk_dict):
            privacy_id, partition_dict = pid_pk_dict
            for partition_key, values in partition_dict.items():
                yield (privacy_id, partition_key), values

        # Unnest the list per privacy id.
        col = backend.flat_map(col, rekey_per_privacy_id_per_partition_key,
                               "Unnest")
        # ((privacy_id, partition_key), [value])

        return backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn after per privacy id contribution bounding")


class SamplingCrossPartitionContributionBounder(ContributionBounder):
    """Bounds the total contributions of a privacy_id. todo

    If a privacy_id contributes more than max_contributions, then
    max_contributions contributions are uniformly sampled, otherwise all
    contributions are kept.
    """

    def bound_contributions(self, col, params, backend, report_generator,
        aggregate_fn):
        col = backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to ((privacy_id), (partition_key, value))")

        col = self._backend.group_by_key(col, "Group by privacy_id")

        # col = backend.sample_fixed_per_key(col, max_contributions,
        #                                    "Sample per privacy_id")
        # report_generator.add_stage(
        #     f"User contribution bounding: randomly selected not "
        #     f"more than {max_contributions} contributions")
        # Convert the per privacy id list into a dict with key as partition_key
        # and values as the list of input values.
        # TODO: extract
        def collect_values_per_partition_key_per_privacy_id(input_list):
            d = collections.defaultdict(list)
            for key, value in input_list:
                d[key].append(value)
            return d

        col = backend.map_values(
            col, collect_values_per_partition_key_per_privacy_id,
            "Group per (privacy_id, partition_key)")

        # (privacy_id, {partition_key: [value]})

        # todo Rekey it into values per privacy id and partition key.
        def rekey_per_privacy_id_per_partition_key(pid_pk_v_dict):
            privacy_id, partition_value_dict = pid_pk_v_dict
            partitions_values = list(partition_value_dict.items())

            # for partition_key, values in partition_dict.items():
            #     yield (privacy_id, partition_key), values

        # Unnest the list per privacy id.
        col = backend.flat_map(col, rekey_per_privacy_id_per_partition_key,
                               "Unnest")
        # ((privacy_id, partition_key), [value])

        return backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn after per privacy id contribution bounding")