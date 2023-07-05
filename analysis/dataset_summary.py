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
""""""

import pipeline_dp
import dataclasses
from typing import Iterable
from enum import Enum


@dataclasses.dataclass
class PublicPartitionsSummary:
    num_dataset_public: int
    num_dataset_non_public: int
    num_empty_public: int


class _PartitionType(Enum):
    DATASET_PUBLIC = 1
    EMPTY_PUBLIC = 2
    DATASET_NONPUBLIC = 3


def compute_public_partitions_summary(col, backend: pipeline_dp.PipelineBackend,
                                      extractors: pipeline_dp.DataExtractors,
                                      public_partitions):
    dataset_partitions = backend.map(col, extractors.partition_extractor,
                                     "Extract partitions")
    # (partition)

    dataset_partitions = backend.distinct(dataset_partitions, "Distinct")
    # (partition)

    dataset_partitions = backend.map(dataset_partitions, lambda x: (x, True),
                                     "Keyd by partition")
    # (partition, is_from_dataset=True)

    public_partitions = backend.map(public_partitions, lambda x: (x, False),
                                    "Keyd by partition")
    # (partition, is_from_dataset = False)

    partitions = backend.flatten([dataset_partitions, public_partitions],
                                 "flatten")
    # (partition, is_from_dataset: bool)

    col = backend.group_by_key(partitions, "Group by Key")

    # (partition, Iterable)

    def process_fn(_, a: Iterable[bool]) -> _PartitionType:
        a = list(a)
        if len(a) == 2:
            return _PartitionType.DATASET_PUBLIC
        if a[0]:
            return _PartitionType.DATASET_NONPUBLIC
        return _PartitionType.EMPTY_PUBLIC

    col = backend.map_tuple(col, process_fn, "Get Partition Type")

    col = backend.count_per_element(col, "Count partition types")

    col = backend.to_list(col, "To list")

    def to_summary(A: list) -> PublicPartitionsSummary:
        num_dataset_public = num_dataset_non_public = num_empty_public = 0
        for type, count in A:
            if type == _PartitionType.DATASET_PUBLIC:
                num_dataset_public = count
            elif type == _PartitionType.DATASET_NONPUBLIC:
                num_dataset_non_public = count
            else:
                num_empty_public = count

        return PublicPartitionsSummary(num_dataset_public,
                                       num_dataset_non_public, num_empty_public)

    return backend.map(col, to_summary, "ToSummary")
