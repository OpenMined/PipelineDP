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
"""Contains utility classes used for specifying DP aggregation parameters, noise types, and norms."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Callable, Union, Optional
import math
import logging


class Metrics(Enum):
    COUNT = 'count'
    PRIVACY_ID_COUNT = 'privacy_id_count'
    SUM = 'sum'
    MEAN = 'mean'
    VARIANCE = 'variance'
    VECTOR_SUM = 'vector_sum'


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
        metrics: A list of metrics to compute.
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to which one
          unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one unit of
          privacy (e.g. a user) can contribute to a partition.
        max_contributions: A bound on the total number of times one unit of privacy
          (e.g., a user) can contribute.
        budget_weight: Relative weight of the privacy budget allocated to this
          aggregation.
        min_value: Lower bound on each value.
        max_value: Upper bound on each value.
        custom_combiners: Warning: experimental@ Combiners for computing custom
          metrics.
        vector_norm_kind: The type of norm. Used only for VECTOR_SUM metric
        calculations.
        vector_max_norm: Bound on each value of a vector. Used only for
         VECTOR_SUM metric calculations.
        vector_size: Number of coordinates in a vector. Used only for VECTOR_SUM
         metric calculations.
        contribution_bounds_already_enforced: assume that the input dataset
         complies with the bounds provided in max_partitions_contributed and
         max_contributions_per_partition. This option can be used if the dataset
         does not contain any identifiers that can be used to enforce
         contribution bounds automatically.
    """
    metrics: Iterable[Metrics]
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    max_contributions: Optional[int] = None
    budget_weight: float = 1
    low: float = None  # deprecated
    high: float = None  # deprecated
    min_value: float = None
    max_value: float = None
    public_partitions: Any = None  # deprecated
    custom_combiners: Iterable['CustomCombiner'] = None
    vector_norm_kind: NormKind = NormKind.Linf
    vector_max_norm: float = None
    vector_size: int = None
    contribution_bounds_already_enforced: bool = False

    def __post_init__(self):
        if self.low is not None:
            raise ValueError(
                "AggregateParams: please use min_value instead of low")
        if self.high is not None:
            raise ValueError(
                "AggregateParams: please use max_value instead of high")
        if self.metrics:
            needs_min_max_value = Metrics.SUM in self.metrics \
                                  or Metrics.MEAN in self.metrics \
                                  or Metrics.VARIANCE in self.metrics
            if needs_min_max_value and (self.min_value is None or
                                        self.max_value is None):
                raise ValueError(
                    "AggregateParams: min_value and max_value must be set")
            if needs_min_max_value and (_not_a_proper_number(self.min_value) or
                                        _not_a_proper_number(self.max_value)):
                raise ValueError(
                    "AggregateParams: min_value and max_value must be both "
                    "finite numbers")
            if needs_min_max_value and self.max_value < self.min_value:
                raise ValueError(
                    "AggregateParams: max_value must be equal to or greater than"
                    " min_value")
            if Metrics.VECTOR_SUM in self.metrics and \
            (Metrics.SUM in self.metrics or \
            Metrics.MEAN in self.metrics or \
            Metrics.VARIANCE in self.metrics):
                raise ValueError(
                    "AggregateParams: vector sum can not be computed together "
                    "with scalar metrics, like sum, mean etc")
            if self.contribution_bounds_already_enforced and Metrics.PRIVACY_ID_COUNT in self.metrics:
                raise ValueError(
                    "AggregateParams: Cannot calculate PRIVACY_ID_COUNT when "
                    "contribution_bounds_already_enforced is set to True.")
        if self.custom_combiners:
            logging.warning("Warning: custom combiners are used. This is an "
                            "experimental feature. It might not work properly "
                            "and it might be changed orremoved without any "
                            "notifications.")
        if self.metrics and self.custom_combiners:
            # TODO(dvadym): after implementation of custom combiners to verify
            # whether this check is required?
            raise ValueError(
                "Custom combiners can not be used with standard metrics")
        if self.public_partitions:
            raise ValueError(
                "AggregateParams.public_partitions is deprecated. Please use public_partitions argument in DPEngine.aggregate insead."
            )
        if self.max_contributions is not None:
            _check_is_positive_int(self.max_contributions, "max_contributions")
            if ((self.max_partitions_contributed is not None) or
                (self.max_contributions_per_partition is not None)):
                raise ValueError(
                    "AggregateParams: only one in max_contributions or "
                    "both max_partitions_contributed and "
                    "max_contributions_per_partition must be set")
        else:  # self.max_contributions is None
            n_not_none = _count_not_none(self.max_partitions_contributed,
                                         self.max_contributions_per_partition)
            if n_not_none == 0:
                raise ValueError(
                    "AggregateParams: either max_contributions must be set or "
                    "both max_partitions_contributed and "
                    "max_contributions_per_partition must be set.")
            elif n_not_none == 1:
                raise ValueError(
                    "AggregateParams: either none or both from "
                    "max_partitions_contributed and "
                    " max_contributions_per_partition must be set.")
            _check_is_positive_int(self.max_partitions_contributed,
                                   "max_partitions_contributed")
            _check_is_positive_int(self.max_contributions_per_partition,
                                   "max_contributions_per_partition")

    def __str__(self):
        if self.custom_combiners:
            return f"Custom combiners: {[c.metrics_names() for c in self.custom_combiners]}"
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

    # TODO: Add support for contribution_bounds_already_enforced

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
        min_value: Lower bound on each value.
        max_value: Upper bound on each value.
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
    public_partitions: Union[Iterable, 'PCollection',
                             'RDD'] = None  # deprecated

    def __post_init__(self):
        if self.low is not None:
            raise ValueError("SumParams: please use min_value instead of low")

        if self.high is not None:
            raise ValueError("SumParams: please use max_value instead of high")

        if self.public_partitions:
            raise ValueError(
                "SumParams.public_partitions is deprecated. Please read API documentation for anonymous Sum transform."
            )


@dataclass
class VectorSumParams:
    """Specifies parameters for differentially-private vector summation calculation.

    Args:
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to which one
          unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one unit of
          privacy (e.g. a user) can contribute to a partition.
              min_value: Lower bound on each value.
        budget_weight: Relative weight of the privacy budget allocated to this
          aggregation.
        vector_norm_kind: The type of norm.
        calculations.
        vector_max_norm: Bound on each value of a vector.
        vector_size: Number of coordinates in a vector.
        partition_extractor: A function which, given an input element, will return its partition id.
        value_extractor: A function which, given an input element, will return its value.
    """
    partition_extractor: Callable
    value_extractor: Callable
    vector_norm_kind: NormKind
    vector_max_norm: float
    vector_size: int
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    budget_weight: float = 1


@dataclass
class VarianceParams:
    """Specifies parameters for differentially-private variance calculation.

    Args:
        noise_kind: Kind of noise to use for the DP calculations.
        max_partitions_contributed: Bounds the number of partitions in which one
            unit of privacy (e.g., a user) can participate.
        max_contributions_per_partition: Bounds the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        min_value: Lower bound on a value contributed by a unit of privacy in a partition.
        max_value: Upper bound on a value contributed by a unit of privacy in a
            partition.
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
    public_partitions: Union[Iterable, 'PCollection',
                             'RDD'] = None  # deprecated

    def __post_init__(self):
        if self.public_partitions:
            raise ValueError(
                "VarianceParams.public_partitions is deprecated. Please read API documentation for anonymous Variance transform."
            )


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
    public_partitions: Union[Iterable, 'PCollection',
                             'RDD'] = None  # deprecated

    def __post_init__(self):
        if self.public_partitions:
            raise ValueError(
                "MeanParams.public_partitions is deprecated. Please read API documentation for anonymous Mean transform."
            )


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

    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    max_contributions_per_partition: int
    partition_extractor: Callable
    budget_weight: float = 1
    public_partitions: Union[Iterable, 'PCollection',
                             'RDD'] = None  # deprecated

    def __post_init__(self):
        if self.public_partitions:
            raise ValueError(
                "CountParams.public_partitions is deprecated. Please read API documentation for anonymous Count transform."
            )


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
    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    partition_extractor: Callable
    budget_weight: float = 1
    public_partitions: Union[Iterable, 'PCollection',
                             'RDD'] = None  # deprecated

    def __post_init__(self):
        if self.public_partitions:
            raise ValueError(
                "PrivacyIdCountParams.public_partitions is deprecated. Please "
                "read API documentation for anonymous PrivacyIdCountParams "
                "transform.")


def _not_a_proper_number(num: Any) -> bool:
    """Returns true if num is inf or NaN, false otherwise."""
    return math.isnan(num) or math.isinf(num)


def _check_is_positive_int(num: Any, field_name: str) -> bool:
    if not (isinstance(num, int) and num > 0):
        raise ValueError(
            f"{field_name} has to be positive integer, but {num} given.")


def _count_not_none(*args):
    return sum([1 for arg in args if arg is not None])
