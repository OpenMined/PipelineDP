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
"""Contains utility classes used for specifying DP aggregation parameters,
noise types, and norms."""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence, Callable, Optional, List

import numpy as np

from pipeline_dp import input_validators


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
        if not isinstance(other, Metric):
            return False
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
        if self.value == NoiseKind.GAUSSIAN.value:
            return MechanismType.GAUSSIAN


class PartitionSelectionStrategy(Enum):
    TRUNCATED_GEOMETRIC = 'Truncated Geometric'
    LAPLACE_THRESHOLDING = 'Laplace Thresholding'
    GAUSSIAN_THRESHOLDING = 'Gaussian Thresholding'

    @property
    def is_thresholding(self) -> bool:
        return self in [self.LAPLACE_THRESHOLDING, self.GAUSSIAN_THRESHOLDING]

    @property
    def mechanism_type(self) -> 'MechanismType':
        if self.value == self.GAUSSIAN_THRESHOLDING.value:
            return MechanismType.GAUSSIAN_THRESHOLDING
        if self.value == self.LAPLACE_THRESHOLDING.value:
            return MechanismType.LAPLACE_THRESHOLDING
        return MechanismType.TRUNCATED_GEOMETRIC


class MechanismType(Enum):
    LAPLACE = 'Laplace'
    GAUSSIAN = 'Gaussian'
    LAPLACE_THRESHOLDING = 'Laplace Thresholding'
    GAUSSIAN_THRESHOLDING = 'Gaussian Thresholding'
    TRUNCATED_GEOMETRIC = 'Truncated Geometric'
    GENERIC = 'Generic'

    def to_noise_kind(self):
        if self.value == MechanismType.LAPLACE.value:
            return NoiseKind.LAPLACE
        if self.value == MechanismType.GAUSSIAN.value:
            return NoiseKind.GAUSSIAN
        if self.value == MechanismType.LAPLACE_THRESHOLDING.value:
            return NoiseKind.LAPLACE
        if self.value == MechanismType.GAUSSIAN_THRESHOLDING.value:
            return NoiseKind.GAUSSIAN
        raise ValueError(f"MechanismType {self.value} can not be converted to "
                         f"NoiseKind")

    def to_partition_selection_strategy(self) -> PartitionSelectionStrategy:
        if self.value == MechanismType.LAPLACE_THRESHOLDING.value:
            return PartitionSelectionStrategy.LAPLACE_THRESHOLDING
        if self.value == MechanismType.GAUSSIAN_THRESHOLDING.value:
            return PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
        raise ValueError(f"MechanismType {self.value} can not be converted to "
                         f"PartitionSelectionStrategy")

    def is_thresholding_mechanism(self):
        return self.value in [
            MechanismType.LAPLACE_THRESHOLDING.value,
            MechanismType.GAUSSIAN_THRESHOLDING.value
        ]


def noise_to_thresholding(noise_kind: NoiseKind) -> MechanismType:
    if noise_kind == NoiseKind.LAPLACE:
        return MechanismType.LAPLACE_THRESHOLDING
    if noise_kind == NoiseKind.GAUSSIAN:
        return MechanismType.GAUSSIAN_THRESHOLDING
    raise ValueError(f"NoiseKind {noise_kind} can not be converted to "
                     f"Thresholding mechanism")


class NormKind(Enum):
    Linf = "linf"
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"


@dataclass
class CalculatePrivateContributionBoundsParams:
    """Specifies parameters for function DPEngine.calculate_private_contribution_bounds()

    WARNING: Aggregation, where the calculated bounds will be used, should be
    one of the following: COUNT, PRIVACY_ID_COUNT.
    Other aggregations (metrics in AggregateParams) are not supported.

    Attributes:
        aggregation_noise_kind: noise that will be used in the aggregation.
        aggregation_eps: epsilon that will be used in the aggregation.
        aggregation_delta: delta that will be used in the aggregation.
        calculation_eps: epsilon that will be used in the computation
          of private contribution bounds.
        max_partitions_contributed_upper_bound: The biggest ever possible value
          for max_partitions_contributed.
    """
    aggregation_noise_kind: NoiseKind
    aggregation_eps: float
    aggregation_delta: float
    calculation_eps: float
    max_partitions_contributed_upper_bound: int

    def __post_init__(self):
        input_validators.validate_epsilon_delta(
            self.aggregation_eps, self.aggregation_delta,
            "CalculatePrivateContributionBoundsParams")
        if self.aggregation_noise_kind is None:
            raise ValueError("aggregation_noise_kind must be set.")
        if (self.aggregation_noise_kind == NoiseKind.GAUSSIAN and
                self.aggregation_delta == 0):
            raise ValueError(
                "The Gaussian noise requires that the aggregation_delta is "
                "greater than 0.")
        input_validators.validate_epsilon_delta(
            self.calculation_eps, 0, "CalculatePrivateContributionBoundsParams")
        _check_is_positive_int(self.max_partitions_contributed_upper_bound,
                               "max_partitions_contributed_upper_bound")


@dataclass
class PrivateContributionBounds:
    """Contribution bounds computed in a differential private way that can be
    used in COUNT and PRIVACY_ID_COUNT aggregations.

    Attributes:
        max_partitions_contributed: A bound on the number of partitions to which
          one unit of privacy (e.g., a user) can contribute
          (also referred to as l_0).
    """
    max_partitions_contributed: int


@dataclass
class AggregateParams:
    """Specifies parameters for function DPEngine.aggregate()

    Attributes:
        metrics: A list of metrics to compute.
        noise_kind: The type of noise to use for the DP calculations.
        max_partitions_contributed: A bound on the number of partitions to
          which one unit of privacy (e.g., a user) can contribute.
        max_contributions_per_partition: A bound on the number of times one
          unit of privacy (e.g. a user) can contribute to a partition.
        max_contributions: A bound on the total number of times one unit of
          privacy (e.g., a user) can contribute.
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
        public_partitions_already_filtered: if true, it is assumed that records
         with partition_key which are not in public_partitions are already
         removed from the dataset. It can only be used with public partitions.
        partition_selection_strategy: which strategy to use for private
         partition selection. It is ignored when public partitions are used.
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with at least pre_threshold privacy units isn't necessarily
         kept. It is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
        perform_cross_partition_contribution_bounding: whether to perform cross
         partition contribution bounding.  
         Warning: turn off cross partition contribution bounding only when the 
         number of contributed partitions per privacy unit is already bounded
         by max_partitions_contributed.
        output_noise_stddev: if True, the output will contain the applied noise
         standard deviation, in form <lower_case_metric_name>_noise_stddev, e.g.
         count_noise_stddev. Currently COUNT, PRIVACY_ID_COUNT, SUM are
         supported.
    """
    metrics: List[Metric]
    noise_kind: NoiseKind = NoiseKind.LAPLACE
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    max_contributions: Optional[int] = None
    budget_weight: float = 1
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_sum_per_partition: Optional[float] = None
    max_sum_per_partition: Optional[float] = None
    custom_combiners: Sequence['CustomCombiner'] = None
    vector_norm_kind: Optional[NormKind] = None
    vector_max_norm: Optional[float] = None
    vector_size: Optional[int] = None
    contribution_bounds_already_enforced: bool = False
    public_partitions_already_filtered: bool = False
    partition_selection_strategy: PartitionSelectionStrategy = PartitionSelectionStrategy.TRUNCATED_GEOMETRIC
    pre_threshold: Optional[int] = None
    post_aggregation_thresholding: bool = False
    perform_cross_partition_contribution_bounding: bool = True
    output_noise_stddev: bool = False

    @property
    def metrics_str(self) -> str:
        if self.custom_combiners:
            return f"custom combiners={[c.metrics_names() for c in self.custom_combiners]}"
        if self.metrics:
            return f"metrics={[str(m) for m in self.metrics]}"
        return "metrics=[]"

    @property
    def bounds_per_contribution_are_set(self) -> bool:
        return self.min_value is not None and self.max_value is not None

    @property
    def bounds_per_partition_are_set(self) -> bool:
        return (self.min_sum_per_partition is not None and
                self.max_sum_per_partition is not None)

    def __post_init__(self):
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
                if (Metrics.SUM in self.metrics or
                        Metrics.MEAN in self.metrics or
                        Metrics.VARIANCE in self.metrics):
                    raise ValueError(
                        "AggregateParams: vector sum can not be computed "
                        "together with scalar metrics such as sum, mean etc")
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

            if (self.contribution_bounds_already_enforced and
                    Metrics.PRIVACY_ID_COUNT in self.metrics):
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
                raise ValueError("AggregateParams: either none or both "
                                 "max_partitions_contributed and "
                                 "max_contributions_per_partition must be set.")
            _check_is_positive_int(self.max_partitions_contributed,
                                   "max_partitions_contributed")
            _check_is_positive_int(self.max_contributions_per_partition,
                                   "max_contributions_per_partition")
        if self.pre_threshold is not None:
            _check_is_positive_int(self.pre_threshold, "pre_threshold")

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
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with at least pre_threshold privacy units isn't necessarily
         kept. It is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
        contribution_bounds_already_enforced: assume that the input dataset
         complies with the bounds provided in max_partitions_contributed and
         max_contributions_per_partition. This option can be used if the dataset
         does not contain any identifiers that can be used to enforce
         contribution bounds automatically.
    """
    max_partitions_contributed: int
    budget_weight: float = 1
    partition_selection_strategy: PartitionSelectionStrategy = PartitionSelectionStrategy.TRUNCATED_GEOMETRIC
    pre_threshold: Optional[int] = None
    contribution_bounds_already_enforced: bool = False

    # TODO: Add support for contribution_bounds_already_enforced

    def __post_init__(self):
        if self.pre_threshold is not None:
            _check_is_positive_int(self.pre_threshold, "pre_threshold")

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
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with pre_threshold privacy units isn't necessarily kept. It
         is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
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
    pre_threshold: Optional[int] = None

    # TODO: add validation in __post_init__


@dataclass
class VarianceParams:
    """Specifies parameters for differentially-private variance calculation.

    Attributes:
        noise_kind: Kind of noise to use for the DP calculations.
        max_partitions_contributed: Bounds the number of partitions in which one
            unit of privacy (e.g., a user) can participate.
        max_contributions_per_partition: Bounds the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        min_value: Lower bound on a value contributed by a unit of privacy in
            a partition.
        max_value: Upper bound on a value contributed by a unit of privacy in a
            partition.
        partition_extractor: A function for partition id extraction from a
            collection record.
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
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with pre_threshold privacy units isn't necessarily kept. It
         is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
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
    pre_threshold: Optional[int] = None

    # TODO: add validation in __post_init__


@dataclass
class MeanParams:
    """Specifies parameters for differentially-private mean calculation.

    Attributes:
        noise_kind: Kind of noise to use for the DP calculations.
        max_partitions_contributed: Bounds the number of partitions in which one
            unit of privacy (e.g., a user) can participate.
        max_contributions_per_partition: Bounds the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        min_value: Lower bound on a value contributed by a unit of privacy in
            a partition.
        max_value: Upper bound on a value contributed by a unit of privacy in a
            partition.
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
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with pre_threshold privacy units isn't necessarily kept. It
         is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
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
    pre_threshold: Optional[int] = None

    # TODO: add validation in __post_init__


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
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with pre_threshold privacy units isn't necessarily kept. It
         is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    max_contributions_per_partition: int
    partition_extractor: Callable
    budget_weight: float = 1
    contribution_bounds_already_enforced: bool = False
    pre_threshold: Optional[int] = None

    # TODO: add validation in __post_init__


@dataclass
class PrivacyIdCountParams:
    """Specifies parameters for differentially-private privacy id count
    calculation.

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
        pre_threshold: the minimum amount of privacy units which are required
         for keeping a partition in private partition selection. Note that this
         is in addition to a differentially private partition selection, so a
         partition with pre_threshold privacy units isn't necessarily kept. It
         is ignored when public partitions are used.
         More details on pre-thresholding are in
         https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
    """

    noise_kind: NoiseKind
    max_partitions_contributed: int
    partition_extractor: Callable
    budget_weight: float = 1
    contribution_bounds_already_enforced: bool = False
    pre_threshold: Optional[int] = None

    # TODO: add validation in __post_init__


@dataclass
class AddDPNoiseParams:
    """Specifies parameters for function DPEngine.add_dp_noise()

    Important: unlike the other methods, this method does not enforce the
    sensitivity by contribution bounding and relies on the caller to ensure the
    provided data satisfies the provided bound.

    The noise in DP is scaled with sensitivity, where sensitivity for query q is
     sensitivity = \max_{D, D' neighbouring} ||q(D)-q(D')||
    Where ||.|| should be a proper norm, L1 for Laplace Mechanism, L2 for
    Gaussian Mechanism.
    See https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm on norm details.

    In our case, q values are represented as collection of
    (partition_key, value). Where partition_key is any identifier, value is
    a scalar. q values can be represented as a vector indexed by partition_key.
    One of the convenient ways to define l1/l2 sensitivity is through l0/linf
    sensitivities.
    Namely

    l1_sensitivity = l0_sensitivity*linf_sensitivity
    l2_sensitivity = sqrt(l0_sensitivity)*linf_sensitivity

    For applying the Laplace mechanism either l1_sensitivity or
    (l0_sensitivity, linf_sensitivity) must be specified.
    
    For applying the Gaussian mechanism either l2_sensitivity or
    (l0_sensitivity, linf_sensitivity) must be specified.

    Attributes:
       noise_kind: The type of noise to use for the DP calculations.
       l0_sensitivity: the maximum number of partition for which 1 privacy unit
         can contribute.
       linf_sensitivity: the maximum difference of values in one partition which
         can achieved by adding or removing one privacy unit from the dataset.
       l1_sensitivity: the sensitivity in L1 norm.
       l2_sensitivity: the sensitivity in L2 norm.
       budget_weight: Relative weight of the privacy budget allocated to this
         aggregation.
       output_noise_stddev: if True, the output will contain the applied noise
         standard deviation, namely the output will be NamedTuple(noisified_value,
         noise_stddev).
   """
    noise_kind: NoiseKind
    l0_sensitivity: Optional[int] = None
    linf_sensitivity: Optional[float] = None
    l1_sensitivity: Optional[float] = None
    l2_sensitivity: Optional[float] = None
    budget_weight: float = 1
    output_noise_stddev: bool = False

    def __post_init__(self):

        def check_is_positive(num: Any, name: str) -> bool:
            if num is not None and num <= 0:
                raise ValueError(f"{name} must be positive, but {num} given.")

        check_is_positive(self.l0_sensitivity, "l0_sensitivity")
        check_is_positive(self.linf_sensitivity, "linf_sensitivity")
        check_is_positive(self.l1_sensitivity, "l1_sensitivity")
        check_is_positive(self.l2_sensitivity, "l2_sensitivity")
        check_is_positive(self.budget_weight, "budget_weight")


def _not_a_proper_number(num: Any) -> bool:
    """Returns true if num is inf or NaN, false otherwise."""
    return math.isnan(num) or math.isinf(num)


def _check_is_positive_int(num: Any, field_name: str) -> None:
    if not (_is_int(num) and num > 0):
        raise ValueError(
            f"{field_name} has to be positive integer, but {num} given.")


def _count_not_none(*args):
    return sum([1 for arg in args if arg is not None])


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer))


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
