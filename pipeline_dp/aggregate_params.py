"""Contains utility classes used for specifying DP aggregation parameters, noise types, and norms."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


class Metrics(Enum):
    COUNT = 'count'
    PRIVACY_ID_COUNT = 'privacy_id_count'
    SUM = 'sum'
    MEAN = 'mean'
    VAR = 'variance'


class NoiseKind(Enum):
    LAPLACE = 'laplace'
    GAUSSIAN = 'gaussian'


class MechanismType(Enum):
    LAPLACE = 'laplace'
    GAUSSIAN = 'gaussian'
    GENERIC = 'generic'


class NormKind(Enum):
    Linf = "linf"
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


@dataclass
class AggregateParams:
    """Specifies parameters for function DPEngine.aggregate()

  Args:
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

    metrics: Iterable[Metrics]
    max_partitions_contributed: int
    max_contributions_per_partition: int
    low: float = None
    high: float = None
    budget_weight: float = 1
    public_partitions: Any = None
