"""Contains utility classes used for specifying DP aggregation parameters, noise types, and norms."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Callable, Union


class Metrics(Enum):
    COUNT = 'count'
    PRIVACY_ID_COUNT = 'privacy_id_count'
    SUM = 'sum'
    MEAN = 'mean'
    VAR = 'variance'


class NoiseKind(Enum):
    LAPLACE = 'laplace'
    GAUSSIAN = 'gaussian'

    def convert_to_mechanism_type(self):
        if self.value == NoiseKind.LAPLACE.value:
            return MechanismType.LAPLACE
        elif self.value == NoiseKind.GAUSSIAN.value:
            return MechanismType.GAUSSIAN


class MechanismType(Enum):
    LAPLACE = 'Laplace'
    GAUSSIAN = 'Gaussian'
    GENERIC = 'Truncated Geometric'


class NormKind(Enum):
    Linf = "linf"
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


@dataclass
class AggregateParams:
    """Specifies parameters for function DPEngine.aggregate()

  Args:
    noise_kind: Kind of noise to use for the DP calculations.
    metrics: Metrics to compute.
    max_partitions_contributed: Bounds the number of partitions in which one
      unit of privacy (e.g., a user) can participate.
    max_contributions_per_partition: Bounds the number of times one unit of
      privacy (e.g. a user) can contribute to a partition.
    low: Lower bound on a value contributed by a unit of privacy in a partition.
    high: Upper bound on a value contributed by a unit of privacy in a
      partition.
    public_partitions: a collection of partition keys that will be present in
      the result.
  """

    noise_kind: NoiseKind
    metrics: Iterable[Metrics]
    max_partitions_contributed: int
    max_contributions_per_partition: int
    budget_weight: float = 1
    low: float = None
    high: float = None
    public_partitions: Any = None

    def __str__(self):
        return f"Metrics: {[m.value for m in self.metrics]}"


# TODO: Think of whether this class should be used for both low-level API
#       (dp_engine) and high-level API (private_spark, private_beam, etc.).
@dataclass
class SelectPrivatePartitionsParams:
    """Specifies parameters for differentially-private partition selection.

    Args:
        max_partitions_contributed: Maximum number of partitions per privacy ID.
            The algorithm will drop contributions over this limit. To keep more
            data, this should be a good estimate of the realistic upper bound.
            Significantly over- or under-estimating this may increase the amount
            of dropped partitions. You can experiment with different values to
            select which one retains more partitions.

    """
    max_partitions_contributed: int

    def __str__(self):
        return "Private Partitions"


@dataclass
class SumParams:
    """Specifies parameters for differentially-private sum calculation.

    Args:
        noise_kind: Kind of noise to use for the DP calculations.
        max_partitions_contributed: Bounds the number of partitions in which one
            unit of privacy (e.g., a user) can participate.
        max_contributions_per_partition: Bounds the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        low: Lower bound on a value contributed by a unit of privacy in a partition.
        high: Upper bound on a value contributed by a unit of privacy in a
            partition.
        public_partitions: A collection of partition keys that will be present in
            the result.
        partition_extractor: A function for partition id extraction from an RDD record.
        value_extractor: A function for extraction of value
            for which the sum will be calculated.
  """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    max_contributions_per_partition: int
    low: float
    high: float
    public_partitions: Union[list, 'PCollection', 'RDD']
    partition_extractor: Callable
    value_extractor: Callable
    budget_weight: float = 1
