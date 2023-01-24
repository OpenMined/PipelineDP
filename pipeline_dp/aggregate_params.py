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
from typing import Any, Iterable, Sequence, Callable, Union, Optional, List
import math
import logging


@dataclass
class Metric:
    """Represents a DP metric.

    Attributes:
        name: the name of the metric, like 'COUNT', 'PERCENTILE'.
        parameter: an optional parameter of the metric, e.g. for 90th
        percentile, parameter = 90.
    """
    name: str
    parameter: Optional[float] = None

    def __eq__(self, other: 'Metric') -> bool:
        return self.name == other.name and self.parameter == other.parameter

    def __str__(self):
        if self.parameter is None:
            return self.name
        return f'{self.name}({self.parameter})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(str(self))

    @property
    def is_percentile(self):
        return self.name == 'PERCENTILE'


class Metrics:
    """Contains all supported DP metrics."""
    COUNT = Metric('COUNT')
    PRIVACY_ID_COUNT = Metric('PRIVACY_ID_COUNT')
    SUM = Metric('SUM')
    MEAN = Metric('MEAN')
    VARIANCE = Metric('VARIANCE')
    VECTOR_SUM = Metric('VECTOR_SUM')

    @classmethod
    def PERCENTILE(cls, percentile_to_compute: float):
        return Metric('PERCENTILE', percentile_to_compute)


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
    GENERIC = 'Generic'


class NormKind(Enum):
    Linf = "linf"
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


class PartitionSelectionStrategy(Enum):
    TRUNCATED_GEOMETRIC = 'Truncated Geometric'
    LAPLACE_THRESHOLDING = 'Laplace Thresholding'
    GAUSSIAN_THRESHOLDING = 'Gaussian Thresholding'


@dataclass
class AggregateParams:
    """Specifies parameters for function DPEngine.aggregate()

    Attributes:
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
        min_sum_per_partition: Lower bound on sum per partition. Used only for
        SUM metric calculations. It can not be set when min_value/max_value is
         set.
        max_sum_per_partition: Upper bound on sum per partition. Used only for
        SUM metric calculations. It can not be set when min_value/max_value is
         set.
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
        partition_selection_strategy: which strategy to use for private
         partition selection. It is ignored when public partitions are used.
    """
    metrics: List[Metric]
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    max_contributions: Optional[int] = None
    budget_weight: float = 1
    low: float = None  # deprecated
    high: float = None  # deprecated
    min_value: float = None
    max_value: float = None
    min_sum_per_partition: float = None
    max_sum_per_partition: float = None
    public_partitions: Any = None  # deprecated
    custom_combiners: Sequence['CustomCombiner'] = None
    vector_norm_kind: Optional[NormKind] = None
    vector_max_norm: Optional[float] = None
    vector_size: Optional[int] = None
    contribution_bounds_already_enforced: bool = False
    partition_selection_strategy: PartitionSelectionStrategy = PartitionSelectionStrategy.TRUNCATED_GEOMETRIC

    @property
    def metrics_str(self) -> str:
        if self.custom_combiners:
            return f"custom combiners={[c.metrics_names() for c in self.custom_combiners]}"
        return f"metrics={[str(m) for m in self.metrics]}"

    @property
    def bounds_per_contribution_are_set(self) -> bool:
        return self.min_value is not None and self.max_value is not None

    @property
    def bounds_per_partition_are_set(self) -> bool:
        return self.min_sum_per_partition is not None and self.max_sum_per_partition is not None

    def __post_init__(self):
        if self.low is not None:
            raise ValueError(
                "AggregateParams: please use min_value instead of low")
        if self.high is not None:
            raise ValueError(
                "AggregateParams: please use max_value instead of high")

        self._check_both_property_set_or_not("min_value", "max_value")
        self._check_both_property_set_or_not("min_sum_per_partition",
                                             "max_sum_per_partition")

        value_bound = self.min_value is not None
        partition_bound = self.min_sum_per_partition is not None

        if value_bound and partition_bound:
            raise ValueError(
                "min_value and min_sum_per_partition can not be both set.")

        if value_bound:
            self._check_range_correctness("min_value", "max_value")

        if partition_bound:
            self._check_range_correctness("min_sum_per_partition",
                                          "max_sum_per_partition")

        if self.metrics:
            if Metrics.VECTOR_SUM in self.metrics:
                if Metrics.SUM in self.metrics or Metrics.MEAN in self.metrics or Metrics.VARIANCE in self.metrics:
                    raise ValueError(
                        "AggregateParams: vector sum can not be computed together"
                        " with scalar metrics such as sum, mean etc")
            elif partition_bound:
                all_allowed_metrics = set(
                    [Metrics.SUM, Metrics.PRIVACY_ID_COUNT, Metrics.COUNT])
                not_allowed_metrics = set(
                    self.metrics).difference(all_allowed_metrics)
                if not_allowed_metrics:
                    raise ValueError(
                        f"AggregateParams: min_sum_per_partition is not "
                        f"compatible with metrics {not_allowed_metrics}. Please"
                        f"use min_value/max_value.")
            elif not partition_bound and not value_bound:
                all_allowed_metrics = set(
                    [Metrics.PRIVACY_ID_COUNT, Metrics.COUNT])
                not_allowed_metrics = set(
                    self.metrics).difference(all_allowed_metrics)
                if not_allowed_metrics:
                    raise ValueError(
                        f"AggregateParams: for metrics {not_allowed_metrics} "
                        f"bounds per partition are required (e.g. min_value,"
                        f"max_value).")

            if self.contribution_bounds_already_enforced and Metrics.PRIVACY_ID_COUNT in self.metrics:
                raise ValueError(
                    "AggregateParams: Cannot calculate PRIVACY_ID_COUNT when "
                    "contribution_bounds_already_enforced is set to True.")
        if self.custom_combiners:
            logging.warning("Warning: custom combiners are used. This is an "
                            "experimental feature. It might not work properly "
                            "and it might be changed or removed without any "
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

    def _check_both_property_set_or_not(self, property1_name: str,
                                        property2_name: str):
        value1 = getattr(self, property1_name)
        value2 = getattr(self, property2_name)
        if (value1 is None) != (value2 is None):
            raise ValueError(
                f"AggregateParams: {property1_name} and {property2_name} should"
                f" be both set or both None.")

    def _check_range_correctness(self, min_property_name: str,
                                 max_property_name: str):
        for property_name in [min_property_name, max_property_name]:
            value = getattr(self, property_name)
            if _not_a_proper_number(value):
                raise ValueError(
                    f"AggregateParams: {property_name} must be a finite number")
        min_value = getattr(self, min_property_name)
        max_value = getattr(self, max_property_name)
        if min_value > max_value:
            raise ValueError(
                f"AggregateParams: {max_property_name} must be equal to or "
                f"greater than {min_property_name}")

    def __str__(self):
        return parameters_to_readable_string(self)


@dataclass
class SelectPartitionsParams:
    """Specifies parameters for differentially-private partition selection.

    Attributes:
        max_partitions_contributed: Maximum number of partitions per privacy ID.
            The algorithm will drop contributions over this limit. To keep more
            data, this should be a good estimate of the realistic upper bound.
            Significantly over- or under-estimating this may increase the number
            of dropped partitions.
        budget_weight: Relative weight of the privacy budget allocated to
            partition selection.
        partition_selection_strategy: which strategy to use for private
         partition selection.
    """
    max_partitions_contributed: int
    budget_weight: float = 1
    partition_selection_strategy: PartitionSelectionStrategy = PartitionSelectionStrategy.TRUNCATED_GEOMETRIC

    # TODO: Add support for contribution_bounds_already_enforced

    def __str__(self):
        return "Private Partitions"


@dataclass
class SumParams:
    """Specifies parameters for differentially-private sum calculation.

    Attributes:
        max_partitions_contributed: A bounds on the number of partitions to
            which one unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one unit
            of privacy (e.g. a user) can contribute to a partition.
        min_value: Lower bound on each value.
        max_value: Upper bound on each value.
        partition_extractor: A function which, given an input element, will
            return its partition id.
        value_extractor: A function which, given an input element, will return
            its value.
        budget_weight: Relative weight of the privacy budget allocated to
            partition selection.
        noise_kind: The type of noise to use for the DP calculations.
        contribution_bounds_already_enforced: assume that the input dataset
            complies with the bounds provided in max_partitions_contributed and
            max_contributions_per_partition. This option can be used if the
            dataset does not contain any identifiers that can be used to enforce
            contribution bounds automatically.
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
    contribution_bounds_already_enforced: bool = False
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
class VarianceParams:
    """Specifies parameters for differentially-private variance calculation.

    Attributes:
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
        budget_weight: Relative weight of the privacy budget allocated to
            partition selection.
        noise_kind: The type of noise to use for the DP calculations.
        contribution_bounds_already_enforced: assume that the input dataset
            complies with the bounds provided in max_partitions_contributed and
            max_contributions_per_partition. This option can be used if the
            dataset does not contain any identifiers that can be used to enforce
            contribution bounds automatically.
    """
    max_partitions_contributed: int
    max_contributions_per_partition: int
    min_value: float
    max_value: float
    partition_extractor: Callable
    value_extractor: Callable
    budget_weight: float = 1
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    contribution_bounds_already_enforced: bool = False
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

    Attributes:
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
        budget_weight: Relative weight of the privacy budget allocated to
            partition selection.
        noise_kind: The type of noise to use for the DP calculations.
        contribution_bounds_already_enforced: assume that the input dataset
            complies with the bounds provided in max_partitions_contributed and
            max_contributions_per_partition. This option can be used if the
            dataset does not contain any identifiers that can be used to enforce
            contribution bounds automatically.
    """
    max_partitions_contributed: int
    max_contributions_per_partition: int
    min_value: float
    max_value: float
    partition_extractor: Callable
    value_extractor: Callable
    budget_weight: float = 1
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    contribution_bounds_already_enforced: bool = False
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

    Attributes:
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to which
            one unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one unit
            of privacy (e.g. a user) can contribute to a partition.
        partition_extractor: A function which, given an input element, will
            return its partition id.
        budget_weight: Relative weight of the privacy budget allocated for this
            operation.
        contribution_bounds_already_enforced: assume that the input dataset
            complies with the bounds provided in max_partitions_contributed and
            max_contributions_per_partition. This option can be used if the
            dataset does not contain any identifiers that can be used to enforce
            contribution bounds automatically.
    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    max_contributions_per_partition: int
    partition_extractor: Callable
    budget_weight: float = 1
    contribution_bounds_already_enforced: bool = False
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

    Attributes:
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to which
            one unit of privacy (e.g., a user) can contribute.
        budget_weight: Relative weight of the privacy budget allocated for this
            operation.
        partition_extractor: A function which, given an input element, will
            return its partition id.
        budget_weight: Relative weight of the privacy budget allocated for this
            operation.
        contribution_bounds_already_enforced: assume that the input dataset
            complies with the bounds provided in max_partitions_contributed and
            max_contributions_per_partition. This option can be used if the
            dataset does not contain any identifiers that can be used to enforce
            contribution bounds automatically.
    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    partition_extractor: Callable
    budget_weight: float = 1
    contribution_bounds_already_enforced: bool = False
    public_partitions: Union[Sequence, 'PCollection',
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


def _add_if_obj_has_property(obj: Any, property_name: str, n_spaces,
                             res: List[str]):
    if not hasattr(obj, property_name):
        return
    value = getattr(obj, property_name)
    if value is None:
        return
    res.append(" " * n_spaces + f"{property_name}={value}")


def parameters_to_readable_string(params,
                                  is_public_partition: Optional[bool] = None
                                 ) -> str:
    result = [f"{type(params).__name__}:"]
    if hasattr(params, "metrics_str"):
        result.append(f" {params.metrics_str}")
    if hasattr(params, "noise_kind"):
        result.append(f" noise_kind={params.noise_kind.value}")
    if hasattr(params, "budget_weight"):
        result.append(f" budget_weight={params.budget_weight}")
    result.append(f" Contribution bounding:")
    _add_if_obj_has_property(params, "max_partitions_contributed", 2, result)
    _add_if_obj_has_property(params, "max_contributions_per_partition", 2,
                             result)
    _add_if_obj_has_property(params, "max_contributions", 2, result)
    _add_if_obj_has_property(params, "min_value", 2, result)
    _add_if_obj_has_property(params, "max_value", 2, result)
    _add_if_obj_has_property(params, "min_sum_per_partition", 2, result)
    _add_if_obj_has_property(params, "max_sum_per_partition", 2, result)
    if hasattr(params, "contribution_bounds_already_enforced"
              ) and params.contribution_bounds_already_enforced:
        result.append("  contribution_bounds_already_enforced=True")
    _add_if_obj_has_property(params, "vector_max_norm", 2, result)
    _add_if_obj_has_property(params, "vector_size", 2, result)
    _add_if_obj_has_property(params, "vector_norm_kind", 2, result)

    if is_public_partition is not None:
        type_str = ("public"
                    if is_public_partition else "private") + " partitions"
        result.append(f" Partition selection: {type_str}")

    return "\n".join(result)
