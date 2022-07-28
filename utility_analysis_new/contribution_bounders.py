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
import collections

from pipeline_dp import contribution_bounders


class SamplingCrossAndPerPartitionContributionBounder(
        contribution_bounders.ContributionBounder):
    """'Bounds' the contribution by privacy_id per and cross partitions.

  Because this is for Utility Analysis, it doesn't actually ensure that
  contribution bounds are enforced. Instead, it keeps track of probabilities
  different data points are kept under per-partition and cross-partition
  contribution bounding.

  Only works for count at the moment.
  """

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """See docstrings for this class and the base class."""
        col = backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to ((privacy_id), (partition_key, value))")
        col = backend.group_by_key(
            col, "Group by key to get (privacy_id, [(partition_key, value)])")
        report_generator.add_stage(
            f"Converting ((privacy_id), (partition_key, value)) to"
            f"(privacy_id, [(partition_key, value)])")

        # (privacy_id, [(partition_key, value)])

        # Convert the per-privacy_id list into a dict with key as partition_key
        # and values as the list of input values.
        def collect_values_per_partition_key_per_privacy_id(input_list):
            d = collections.defaultdict(list)
            for key, value in input_list:
                d[key].append(value)
            return d

        col = backend.map_values(
            col, collect_values_per_partition_key_per_privacy_id,
            "Group per (privacy_id, partition_key)")

        # Rekey by (privacy_id, partition_key) and unnest values along with the
        # number of partitions contributed per privacy_id.
        def rekey_per_privacy_id_per_partition_key_and_unnest(pid_pk_dict):
            privacy_id, partition_dict = pid_pk_dict
            num_partitions_contributed = len(partition_dict)
            for partition_key, values in partition_dict.items():
                yield (privacy_id, partition_key), (values,
                                                    num_partitions_contributed)

        # Unnest the list per privacy id.
        col = backend.flat_map(
            col, rekey_per_privacy_id_per_partition_key_and_unnest, "Unnest")

        return backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn with num_partitions_contributed as input")
