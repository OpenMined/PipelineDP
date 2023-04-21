import dataclasses
from typing import Callable


@dataclasses.dataclass
class DataExtractors:
    """Data extractors.

    A set of functions that, given a piece of input, return the privacy id,
    partition key, and value respectively.
    """

    privacy_id_extractor: Callable = None
    partition_extractor: Callable = None
    value_extractor: Callable = None


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
          data, and returns (count, sum, n_partitions).
    """
    partition_extractor: Callable
    preaggregate_extractor: Callable
