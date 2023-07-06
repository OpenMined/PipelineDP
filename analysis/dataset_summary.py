# Copyright 2023 OpenMined.
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
"""Contains dataset summary and the computation of the summary."""

import pipeline_dp
import dataclasses
from typing import Iterable


@dataclasses.dataclass
class PublicPartitionsSummary:
    num_dataset_public_partitions: int
    num_dataset_non_public_partitions: int
    num_empty_public_partitions: int


_DATASET_PUBLIC = 1
_EMPTY_PUBLIC = 2
_DATASET_NONPUBLIC = 3


def compute_public_partitions_summary(col, backend: pipeline_dp.PipelineBackend,
                                      extractors: pipeline_dp.DataExtractors,
                                      public_partitions):
    """Computes Public Partitions Summary from dataset and public partitions.

    Args:
        col: the raw dataset. The collection where all elements are of the same
            type.
        backend: pipeline backend which corresponds to the type of 'col'.
        extractors: functions that extract needed pieces of information
            from elements of 'col'.
        public_partitions: a collection of partition keys that will be present
          in the result. If not provided, partitions will be selected in a DP
          manner.

    Returns:
         1 element collection, which contains a PublicPartitionsSummary object.
    """
    dataset_partitions = backend.map(col, extractors.partition_extractor,
                                     "Extract partitions")
    # (partition)

    dataset_partitions = backend.distinct(dataset_partitions, "Distinct")
    # (partition)

    dataset_partitions = backend.map(dataset_partitions, lambda x: (x, True),
                                     "Keyed by partition")
    # (partition, is_from_dataset=True)

    public_partitions = backend.map(public_partitions, lambda x: (x, False),
                                    "Keyed by partition")
    # (partition, is_from_dataset = False)

    partitions = backend.flatten([dataset_partitions, public_partitions],
                                 "flatten")
    # (partition, is_from_dataset: bool)

    col = backend.group_by_key(partitions, "Group by Key")

    # (partition, Iterable)

    def process_fn(_, a: Iterable[bool]) -> int:
        # a contains up to 2 booleans.
        # True means that the partition is dataset.
        # False means that the partition is in public partitions.
        a = list(a)
        if len(a) == 2:
            return _DATASET_PUBLIC
        if a[0]:
            return _DATASET_NONPUBLIC
        return _EMPTY_PUBLIC

    col = backend.map_tuple(col, process_fn, "Get Partition Type")
    # (partition_type:int)

    col = backend.count_per_element(col, "Count partition types")
    # (partition_type:int, count_partition_type:int)

    col = backend.to_list(col, "To list")

    # 1 element with list of tuples (partition_type, count_partition_type)

    def to_summary(partition_types_count: list) -> PublicPartitionsSummary:
        num_dataset_public = num_dataset_non_public = num_empty_public = 0
        for type, count in partition_types_count:
            if type == _DATASET_PUBLIC:
                num_dataset_public = count
            elif type == _DATASET_NONPUBLIC:
                num_dataset_non_public = count
            else:
                num_empty_public = count

        return PublicPartitionsSummary(num_dataset_public,
                                       num_dataset_non_public, num_empty_public)

    return backend.map(col, to_summary, "ToSummary")
