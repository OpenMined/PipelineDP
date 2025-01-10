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
"""ContributionBounder for utility analysis."""

import numpy as np
from pipeline_dp import contribution_bounders
from pipeline_dp import sampling_utils
from typing import Iterable, Union


class L0LinfAnalysisContributionBounder(
        contribution_bounders.ContributionBounder):
    """'Bounds' the contribution by privacy_id per and cross partitions.

    Because this is for Utility Analysis, it doesn't actually ensure that
    contribution bounds are enforced. Instead, it keeps track the information
    needed for computing impact of contribution bounding.

    If partitions_sampling_prob < 1.0, partitions are subsampled. This sampling
    is deterministic and depends on partition key.
    """

    def __init__(self,
                 partitions_sampling_prob: float,
                 perform_cross_partition_contribution_bounding: bool = True):
        super().__init__()
        self._sampling_probability = partitions_sampling_prob

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """See docstrings for this class and the base class."""

        col = backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to (privacy_id, (partition_key, value))")
        col = backend.group_by_key(
            col, "Group by key to get (privacy_id, [(partition_key, value)])")
        # (privacy_id, [(partition_key, value)])

        col = contribution_bounders.collect_values_per_partition_key_per_privacy_id(
            col, backend)

        # (privacy_id, [(partition_key, [value])])

        # Rekey by (privacy_id, partition_key) and unnest values along with the
        # number of partitions contributed per privacy_id.
        # Sample by partition key if sampling_prob < 1.
        sampler = sampling_utils.ValueSampler(
            self._sampling_probability
        ) if self._sampling_probability < 1 else None

        def rekey_per_privacy_id_per_partition_key_and_unnest(pid_pk_v_values):
            privacy_id, partition_values = pid_pk_v_values
            num_partitions_contributed = len(partition_values)
            num_contributions = sum(
                (len(values) for _, values in partition_values))
            for partition_key, values in partition_values:
                if sampler is not None and not sampler.keep(partition_key):
                    continue
                sum_values = _sum_values(values)

                yield (privacy_id, partition_key), (
                    len(values),
                    sum_values,
                    num_partitions_contributed,
                    num_contributions,
                )

        # Unnest the list per privacy id.
        col = backend.flat_map(
            col, rekey_per_privacy_id_per_partition_key_and_unnest,
            "Unnest per-privacy_id")

        return backend.map_values(col, aggregate_fn, "Apply aggregate_fn")


class LinfAnalysisContributionBounder(contribution_bounders.ContributionBounder
                                     ):
    """'Bounds' the contribution by privacy_id per partitions.

    Because this is for Utility Analysis, it doesn't actually ensure that
    contribution bounds are enforced. Instead, it keeps track the information
    needed for computing impact of contribution bounding.

    If partitions_sampling_prob < 1.0, partitions are subsampled. This sampling
    is deterministic and depends on partition key.
    """

    def __init__(self, partitions_sampling_prob: float):
        super().__init__()
        self._sampling_probability = partitions_sampling_prob

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """See docstrings for this class and the base class."""
        # Sample by partition key if sampling_prob < 1.
        if self._sampling_probability < 1:
            sampler = sampling_utils.ValueSampler(self._sampling_probability)
            col = backend.filter(col, lambda pid, pk, v: sampler.keep(pk),
                                 "Sample partitions")

        col = backend.map_tuple(
            col, lambda pid, pk, v: ((pid, pk), v),
            "Rekey to ((privacy_id, partition_key), value)")
        col = backend.group_by_key(
            col, "Group by key to get ((privacy_id, partition_key), [value])")

        # ((privacy_id, partition_key), [value])

        # Rekey by (privacy_id, partition_key) and unnest values along with the
        # number of partitions contributed per privacy_id.
        def rekey_per_privacy_id_per_partition_key_and_unnest(pid_pk_v_values):
            (privacy_id, partition_key), values = pid_pk_v_values
            num_partitions_contributed = 1
            num_contributions = len(values)
            sum_values = _sum_values(values)
            yield (privacy_id, partition_key), (
                len(values),
                sum_values,
                num_partitions_contributed,
                num_contributions,
            )

        # Unnest the list per privacy id.
        col = backend.flat_map(
            col, rekey_per_privacy_id_per_partition_key_and_unnest,
            "Unnest per-privacy_id")

        return backend.map_values(col, aggregate_fn, "Apply aggregate_fn")


def _sum_values(
    values: Union[Iterable[float], Iterable[Iterable[float]]]
) -> Union[float, tuple[float]]:
    """Sums values"""
    # Sum values.
    # values can contain multi-columns, the format is the following
    # 1 column:
    #   input: values = [v_0:float, ... ]
    #   output: v_0 + ....
    # k columns (k > 1):
    #   input: values = [v_0=(v_00, ... v_0(k-1)), ...]
    #   output: (v_00+v_10+..., ...)
    if not values:
        # Empty public partitions
        return 0
    if len(values) == 1:
        # No need to sum, return 0th value
        return values[0]
    if not isinstance(values[0], Iterable):
        # 1 column
        return sum(values)
    # multiple value columns, sum each column independently
    return tuple(np.array(values).sum(axis=0).tolist())


class NoOpContributionBounder(contribution_bounders.ContributionBounder):

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """Returns a collection with element in the correct format."""
        return backend.map_tuple(
            col, lambda pid, pk, val: ((pid, pk), aggregate_fn(val)),
            "Apply aggregate_fn")
        # ((privacy_id, partition_key), accumulator)
