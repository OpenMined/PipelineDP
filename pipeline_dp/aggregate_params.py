"""Contains utility classes used for specifying DP aggregation parameters, noise types, and norms."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Callable, Union
import math


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
    noise_kind: The type of noise to use for the DP calculations.
    metrics: A list of metrics to compute.
    max_partitions_contributed: A bound on the number of partitions to which one
      unit of privacy (e.g., a user) can contribute.
    max_contributions_per_partition: A bound on the number of times one unit of
      privacy (e.g. a user) can contribute to a partition.
    budget_weight: Relative weight of the privacy budget allocated to this
      aggregation.
    min_value: Lower bound on each value.
    max_value: Upper bound on each value.
    public_partitions: A collection of partition keys that will be present in
      the result. Optional. If not provided, partitions will be selected in a DP
      manner.
  """

    noise_kind: NoiseKind
    metrics: Iterable[Metrics]
    max_partitions_contributed: int
    max_contributions_per_partition: int
    budget_weight: float = 1
    low: float = None  # deprecated
    high: float = None  # deprecated
    min_value: float = None
    max_value: float = None
    public_partitions: Any = None

    def __post_init__(self):
        if self.low is not None:
            raise ValueError(
                "AggregateParams: please use min_value instead of low")
        if self.high is not None:
            raise ValueError(
                "AggregateParams: please use max_value instead of high")
        needs_min_max_value = Metrics.SUM in self.metrics
        if not isinstance(self.max_partitions_contributed,
                          int) or self.max_partitions_contributed <= 0:
            raise ValueError("params.max_partitions_contributed must be set "
                             "to a positive integer")
        if not isinstance(self.max_contributions_per_partition,
                          int) or self.max_contributions_per_partition <= 0:
            raise ValueError(
                "params.max_contributions_per_partition must be set "
                "to a positive integer")
        if needs_min_max_value and (self.min_value is None or
                                    self.max_value is None):
            raise ValueError(
                "params.min_value and params.max_value must be set")
        if needs_min_max_value and (_not_a_proper_number(self.min_value) or
                                    _not_a_proper_number(self.max_value)):
            raise ValueError(
                "params.min_value and params.max_value must be both finite numbers"
            )
        if needs_min_max_value and self.max_value < self.min_value:
            raise ValueError(
                "params.max_value must be equal to or greater than params.min_value"
            )

    def __str__(self):
        return f"Metrics: {[m.value for m in self.metrics]}"


@dataclass
class SelectPartitionsParams:
    """Specifies parameters for differentially-private partition selection.

    Args:
        max_partitions_contributed: Maximum number of partitions per privacy ID.
            The algorithm will drop contributions over this limit. To keep more
            data, this should be a good estimate of the realistic upper bound.
            Significantly over- or under-estimating this may increase the number
            of dropped partitions.
        budget_weight: Relative weight of the privacy budget allocated to
            partition selection.

    """
    max_partitions_contributed: int
    budget_weight: float = 1

    def __str__(self):
        return "Private Partitions"


@dataclass
class SumParams:
    """Specifies parameters for differentially-private sum calculation.

    Args:
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bounds on the number of partitions to which one
            unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        low: Lower bound on each value.
        high: Upper bound on each value.
        public_partitions: A collection of partition keys that will be present in
            the result. Optioanl.
        partition_extractor: A function which, given an input element, will return its partition id.
        value_extractor: A function which, given an input element, will return its value.
  """
    max_partitions_contributed: int
    max_contributions_per_partition: int
    min_value: float
    max_value: float
    partition_extractor: Callable
    value_extractor: Callable
    low: float = None  # deprecated
    high: float = None  # deprecated
    budget_weight: float = 1
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    public_partitions: Union[Iterable, 'PCollection', 'RDD'] = None

    def __post_init__(self):
        if self.low is not None:
            raise ValueError("SumParams: please use min_value instead of low")

        if self.high is not None:
            raise ValueError("SumParams: please use max_value instead of high")


@dataclass
class MeanParams:
    """Specifies parameters for differentially-private mean calculation.

    Args:
        noise_kind: Kind of noise to use for the DP calculations.
        max_partitions_contributed: Bounds the number of partitions in which one
            unit of privacy (e.g., a user) can participate.
        max_contributions_per_partition: Bounds the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        min_value: Lower bound on a value contributed by a unit of privacy in a partition.
        max_value: Upper bound on a value contributed by a unit of privacy in a
            partition.
        public_partitions: A collection of partition keys that will be present in
            the result.
        partition_extractor: A function for partition id extraction from a collection record.
        value_extractor: A function for extraction of value
            for which the sum will be calculated.
  """
    max_partitions_contributed: int
    max_contributions_per_partition: int
    min_value: float
    max_value: float
    partition_extractor: Callable
    value_extractor: Callable
    budget_weight: float = 1
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    public_partitions: Union[Iterable, 'PCollection', 'RDD'] = None


@dataclass
class CountParams:
    """Specifies parameters for differentially-private count calculation.

    Args:
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to which one
            unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        partition_extractor: A function which, given an input element, will return its partition id.
        budget_weight: Relative weight of the privacy budget allocated for this
            operation.
        public_partitions: A collection of partition keys that will be present in
            the result. Optional.

    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    max_contributions_per_partition: int
    partition_extractor: Callable
    budget_weight: float = 1
    public_partitions: Union[Iterable, 'PCollection', 'RDD'] = None


@dataclass
class PrivacyIdCountParams:
    """Specifies parameters for differentially-private privacy id count calculation.

    Args:
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to which one
            unit of privacy (e.g., a user) can contribute.
        budget_weight: Relative weight of the privacy budget allocated for this
            operation.
        partition_extractor: A function which, given an input element, will return its partition id.
        public_partitions: A collection of partition keys that will be present in
            the result. Optional.
    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    partition_extractor: Callable
    budget_weight: float = 1
    public_partitions: Union[Iterable, 'PCollection', 'RDD'] = None


def _not_a_proper_number(num):
    """
    Returns:
        true if num is inf or NaN, false otherwise.
    """
    return math.isnan(num) or math.isinf(num)
