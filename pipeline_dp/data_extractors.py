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
"""Classes for keeping data (privacy unit, partition etc) extractors."""

import dataclasses
from typing import Callable, List, Optional


@dataclasses.dataclass
class DataExtractors:
    """Data extractors.

    A set of functions that, given a piece of input, return the privacy id,
    partition key, and value respectively.
    """

    privacy_id_extractor: Optional[Callable] = None
    partition_extractor: Optional[Callable] = None
    value_extractor: Optional[Callable] = None


@dataclasses.dataclass
class MultiValueDataExtractors(DataExtractors):
    """Data extractors with multiple value extractors.

    Each extractor corresponds to the different value to aggregate.
    """
    value_extractors: Optional[List[Callable]] = None

    def __post_init__(self):
        if self.value_extractors is not None:
            self.value_extractor = lambda row: tuple(
                e(row) for e in self.value_extractors)


@dataclasses.dataclass
class PreAggregateExtractors:
    """Data extractors for pre-aggregated data.

    Pre-aggregated data assumptions: each row corresponds to each
    (privacy_id, partition_key) which is present in the original dataset.
    Each row has
      1. count and sum, which correspond to count and sum of values
    contributed by the privacy_id to the partition_key.
      2. n_partitions, which is the number of partitions contributed by
    privacy_id.

    Attributes:
        partition_extractor: a callable, that takes a row of preaggraged data,
          and returns partition key.
        preaggregate_extractor: a callable, that takes a row of preaggraged
          data, and returns (count, sum, n_partitions, n_contributions).
    """
    partition_extractor: Callable
    preaggregate_extractor: Callable
