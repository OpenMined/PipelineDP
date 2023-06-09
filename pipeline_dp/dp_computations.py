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
"""Differential privacy computing of count, sum, mean, variance."""

import abc
import math
import typing

import numpy as np
from typing import Any, Optional, Tuple, Union

import pipeline_dp
from dataclasses import dataclass
from pydp.algorithms import numerical_mechanisms as dp_mechanisms


@dataclass
class ScalarNoiseParams:
    """The parameters used for computing the dp sum, count, mean, variance."""

    eps: float
    delta: float
    min_value: Optional[float]
    max_value: Optional[float]
    min_sum_per_partition: Optional[float]
    max_sum_per_partition: Optional[float]
    max_partitions_contributed: int
    max_contributions_per_partition: Optional[int]
    noise_kind: pipeline_dp.NoiseKind  # Laplace or Gaussian

    def __post_init__(self):
        assert (self.min_value is None) == (
            self.max_value is None
        ), "min_value and max_value should be or both set or both None."
        assert (self.min_sum_per_partition is None) == (
            self.max_sum_per_partition is None
        ), "min_sum_per_partition and max_sum_per_partition should be or both set or both None."

    def l0_sensitivity(self) -> int:
        """"Returns the L0 sensitivity of the parameters."""
        return self.max_partitions_contributed

    @property
    def bounds_per_contribution_are_set(self) -> bool:
        return self.min_value is not None and self.max_value is not None

    @property
    def bounds_per_partition_are_set(self) -> bool:
        return self.min_sum_per_partition is not None and self.max_sum_per_partition is not None


def compute_squares_interval(min_value: float,
                             max_value: float) -> Tuple[float, float]:
    """Returns the bounds of the interval [min_value^2, max_value^2]."""
    if min_value < 0 < max_value:
        return 0, max(min_value**2, max_value**2)
    return min_value**2, max_value**2


def compute_middle(min_value: float, max_value: float) -> float:
    """"Returns the middle point of the interval [min_value, max_value]."""
    # (min_value + max_value) / 2 may cause an overflow or loss of precision if
    # min_value and max_value are large.
    return min_value + (max_value - min_value) / 2


def compute_l1_sensitivity(l0_sensitivity: float,
                           linf_sensitivity: float) -> float:
    """Calculates the L1 sensitivity based on the L0 and Linf sensitivities.

    Args:
        l0_sensitivity: The L0 sensitivity.
        linf_sensitivity: The Linf sensitivity.

    Returns:
        The L1 sensitivity.
    """
    return l0_sensitivity * linf_sensitivity


def compute_l2_sensitivity(l0_sensitivity: float,
                           linf_sensitivity: float) -> float:
    """Calculates the L2 sensitivity based on the L0 and Linf sensitivities.

    Args:
        l0_sensitivity: The L0 sensitivity.
        linf_sensitivity: The Linf sensitivity.

    Returns:
        The L2 sensitivity.
    """
    return np.sqrt(l0_sensitivity) * linf_sensitivity


def compute_sigma(eps: float, delta: float, l2_sensitivity: float) -> float:
    """Returns the optimal value of sigma for the Gaussian mechanism.

    Args:
        eps: The epsilon value.
        delta: The delta value.
        l2_sensitivity: The L2 sensitivity.
    """
    # TODO: use named arguments, when argument names are added in PyDP on PR
    # https://github.com/OpenMined/PyDP/pull/398.
    return dp_mechanisms.GaussianMechanism(eps, delta, l2_sensitivity).std


def apply_laplace_mechanism(value: float, eps: float, l1_sensitivity: float):
    """Applies the Laplace mechanism to the value.

    Args:
        value: The initial value.
        eps: The epsilon value.
        l1_sensitivity: The L1 sensitivity.

    Returns:
        The value resulted after adding the noise.
    """
    mechanism = dp_mechanisms.LaplaceMechanism(epsilon=eps,
                                               sensitivity=l1_sensitivity)
    return mechanism.add_noise(1.0 * value)


def apply_gaussian_mechanism(value: float, eps: float, delta: float,
                             l2_sensitivity: float):
    """Applies the Gaussian mechanism to the value.

    Args:
        value: The initial value.
        eps: The epsilon value.
        delta: The delta value.
        l2_sensitivity: The L2 sensitivity.

    Returns:
        The value resulted after adding the noise.
    """
    # TODO: use named arguments, when argument names are added in PyDP on PR
    # https://github.com/OpenMined/PyDP/pull/398.
    mechanism = dp_mechanisms.GaussianMechanism(eps, delta, l2_sensitivity)
    return mechanism.add_noise(1.0 * value)


def _add_random_noise(
    value: float,
    eps: float,
    delta: float,
    l0_sensitivity: float,
    linf_sensitivity: float,
    noise_kind: pipeline_dp.NoiseKind,
) -> float:
    """Adds random noise according to the parameters.

    Args:
        value: The initial value.
        eps: The epsilon value.
        delta: The delta value.
        l0_sensitivity: The L0 sensitivity.
        linf_sensitivity: The Linf sensitivity.
        noise_kind: The kind of noise used.

    Returns:
        The value resulted after adding the random noise.
    """
    if noise_kind == pipeline_dp.NoiseKind.LAPLACE:
        l1_sensitivity = compute_l1_sensitivity(l0_sensitivity,
                                                linf_sensitivity)
        return apply_laplace_mechanism(value, eps, l1_sensitivity)
    if noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
        l2_sensitivity = compute_l2_sensitivity(l0_sensitivity,
                                                linf_sensitivity)
        return apply_gaussian_mechanism(value, eps, delta, l2_sensitivity)
    raise ValueError("Noise kind must be either Laplace or Gaussian.")


@dataclass
class AdditiveVectorNoiseParams:
    eps_per_coordinate: float
    delta_per_coordinate: float
    max_norm: float
    l0_sensitivity: float
    linf_sensitivity: float
    norm_kind: pipeline_dp.NormKind
    noise_kind: pipeline_dp.NoiseKind


def _clip_vector(vec: np.ndarray, max_norm: float,
                 norm_kind: pipeline_dp.aggregate_params.NormKind):
    norm_kind = norm_kind.value  # type: str
    if norm_kind == "linf":
        return np.clip(vec, -max_norm, max_norm)
    if norm_kind in {"l1", "l2"}:
        norm_kind = int(norm_kind[-1])
        vec_norm = np.linalg.norm(vec, ord=norm_kind)
        mul_coef = min(1, max_norm / vec_norm)
        return vec * mul_coef
    raise NotImplementedError(
        f"Vector Norm of kind '{norm_kind}' is not supported.")


def add_noise_vector(vec: np.ndarray, noise_params: AdditiveVectorNoiseParams):
    """Adds noise to vector to make the vector sum computation (eps, delta)-DP.

    Args:
        vec: the queried raw vector
        noise_params: parameters of the noise to add to the computation
    """
    vec = _clip_vector(vec, noise_params.max_norm, noise_params.norm_kind)
    vec = np.array([
        _add_random_noise(
            s,
            noise_params.eps_per_coordinate,
            noise_params.delta_per_coordinate,
            noise_params.l0_sensitivity,
            noise_params.linf_sensitivity,
            noise_params.noise_kind,
        ) for s in vec
    ])
    return vec


def equally_split_budget(eps: float, delta: float, no_mechanisms: int):
    """Equally splits the budget (eps, delta) between a given number of mechanisms.

    Args:
        eps, delta: The available budget.
        no_mechanisms: The number of mechanisms between which we split the budget.

    Raises:
        ValueError: The number of mechanisms must be a positive integer.

    Returns:
        An array with the split budgets.
    """
    if no_mechanisms <= 0:
        raise ValueError("The number of mechanisms must be a positive integer.")

    # These variables are used to keep track of the budget used.
    # In this way, we can improve accuracy of floating-point operations.
    eps_used = delta_used = 0
    budgets = []

    for _ in range(no_mechanisms - 1):
        budget = (eps / no_mechanisms, delta / no_mechanisms)
        eps_used += budget[0]
        delta_used += budget[1]
        budgets.append(budget)

    budgets.append((eps - eps_used, delta - delta_used))
    return budgets


def _compute_mean_for_normalized_sum(
    dp_count: float,
    sum: float,
    min_value: float,
    max_value: float,
    eps: float,
    delta: float,
    l0_sensitivity: float,
    max_contributions_per_partition: float,
    noise_kind: pipeline_dp.NoiseKind,
):
    """Helper function to compute the DP mean of a raw sum using the DP count.

    Args:
        dp_count: DP count.
        sum: Non-DP normalized sum.
        min_value, max_value: The lowest/highest contribution of the non-normalized values.
        eps, delta: The budget allocated.
        l0_sensitivity: The L0 sensitivity.
        max_contributions_per_partition: The maximum number of contributions
            per partition.
        noise_kind: The kind of noise used.

    Raises:
        ValueError: The noise kind is invalid.

    Returns:
        The anonymized mean.
    """
    if min_value == max_value:
        return min_value
    middle = compute_middle(min_value, max_value)
    linf_sensitivity = max_contributions_per_partition * abs(middle - min_value)

    dp_normalized_sum = _add_random_noise(sum, eps, delta, l0_sensitivity,
                                          linf_sensitivity, noise_kind)
    # Clamps dp_count to 1.0. We know that actual count > 1 except when the
    # input set is empty, in which case it shouldn't matter much what the
    # denominator is.
    dp_count_clamped = max(1.0, dp_count)
    return dp_normalized_sum / dp_count_clamped


def compute_dp_var(count: int, normalized_sum: float,
                   normalized_sum_squares: float, dp_params: ScalarNoiseParams):
    """Computes DP variance.

    Args:
        count: Non-DP count.
        normalized_sum: Non-DP normalized sum.
        normalized_sum_squares: Non-DP normalized sum of squares.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.

    Returns:
        The tuple of anonymized count, sum, mean and variance.
    """
    # Splits the budget equally between the three mechanisms.
    (
        (count_eps, count_delta),
        (sum_eps, sum_delta),
        (sum_squares_eps, sum_squares_delta),
    ) = equally_split_budget(dp_params.eps, dp_params.delta, 3)
    l0_sensitivity = dp_params.l0_sensitivity()

    dp_count = _add_random_noise(
        count,
        count_eps,
        count_delta,
        l0_sensitivity,
        dp_params.max_contributions_per_partition,
        dp_params.noise_kind,
    )

    # Computes and adds noise to the mean.
    dp_mean = _compute_mean_for_normalized_sum(
        dp_count,
        normalized_sum,
        dp_params.min_value,
        dp_params.max_value,
        sum_eps,
        sum_delta,
        l0_sensitivity,
        dp_params.max_contributions_per_partition,
        dp_params.noise_kind,
    )

    squares_min_value, squares_max_value = compute_squares_interval(
        dp_params.min_value, dp_params.max_value)

    # Computes and adds noise to the mean of squares.
    dp_mean_squares = _compute_mean_for_normalized_sum(
        dp_count, normalized_sum_squares, squares_min_value, squares_max_value,
        sum_squares_eps, sum_squares_delta, l0_sensitivity,
        dp_params.max_contributions_per_partition, dp_params.noise_kind)

    dp_var = dp_mean_squares - dp_mean**2
    if dp_params.min_value != dp_params.max_value:
        dp_mean += compute_middle(dp_params.min_value, dp_params.max_value)

    return dp_count, dp_mean * dp_count, dp_mean, dp_var


def _compute_noise_std(linf_sensitivity: float,
                       dp_params: ScalarNoiseParams) -> float:
    """Computes noise standard deviation using the specified linf sensitivity."""
    if dp_params.noise_kind == pipeline_dp.NoiseKind.LAPLACE:
        l1_sensitivity = compute_l1_sensitivity(dp_params.l0_sensitivity(),
                                                linf_sensitivity)
        mechanism = dp_mechanisms.LaplaceMechanism(epsilon=dp_params.eps,
                                                   sensitivity=l1_sensitivity)
        return mechanism.diversity * np.sqrt(2)
    if dp_params.noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
        l2_sensitivity = compute_l2_sensitivity(dp_params.l0_sensitivity(),
                                                linf_sensitivity)
        return compute_sigma(dp_params.eps, dp_params.delta, l2_sensitivity)
    assert "Only Laplace and Gaussian noise is supported."


def compute_dp_count_noise_std(dp_params: ScalarNoiseParams) -> float:
    """Computes noise standard deviation for DP count."""
    linf_sensitivity = dp_params.max_contributions_per_partition
    return _compute_noise_std(linf_sensitivity, dp_params)


def compute_dp_sum_noise_std(dp_params: ScalarNoiseParams) -> float:
    """Computes noise standard deviation for DP sum."""
    linf_sensitivity = max(abs(dp_params.min_sum_per_partition),
                           abs(dp_params.max_sum_per_partition))
    return _compute_noise_std(linf_sensitivity, dp_params)


class AdditiveMechanism(abc.ABC):
    """Base class for addition DP mechanism (like Laplace of Gaussian)."""

    @abc.abstractmethod
    def add_noise(self, value: Union[int, float]) -> float:
        """Anonymizes value by adding noise."""

    @property
    @abc.abstractmethod
    def noise_kind(self) -> pipeline_dp.NoiseKind:
        pass

    @property
    @abc.abstractmethod
    def noise_parameter(self) -> float:
        """Noise distribution parameter."""
        pass

    @property
    @abc.abstractmethod
    def std(self) -> float:
        """Noise distribution standard deviation."""

    @property
    @abc.abstractmethod
    def sensitivity(self) -> float:
        """Mechanism sensitivity."""

    @abc.abstractmethod
    def describe(self) -> float:
        """Mechanism description for explain computation reports."""


class LaplaceMechanism(AdditiveMechanism):

    def __init__(self, epsilon: float, l1_sensitivity: float):
        self._mechanism = dp_mechanisms.LaplaceMechanism(
            epsilon=epsilon, sensitivity=l1_sensitivity)

    def add_noise(self, value: Union[int, float]) -> float:
        return self._mechanism.add_noise(1.0 * value)

    @property
    def noise_parameter(self) -> float:
        return self._mechanism.diversity

    @property
    def std(self) -> float:
        return self.noise_parameter * math.sqrt(2)

    @property
    def noise_kind(self) -> pipeline_dp.NoiseKind:
        return pipeline_dp.NoiseKind.LAPLACE

    @property
    def sensitivity(self) -> float:
        return self._mechanism.sensitivity

    def describe(self) -> str:
        return (f"Laplace mechanism:  parameter={self.noise_parameter}  eps="
                f"{self._mechanism.epsilon}  l1_sensitivity={self.sensitivity}")


class GaussianMechanism(AdditiveMechanism):

    def __init__(self, epsilon: float, delta: float, l2_sensitivity: float):
        self._l2_sensitivity = l2_sensitivity
        self._mechanism = dp_mechanisms.GaussianMechanism(
            epsilon=epsilon, delta=delta, sensitivity=l2_sensitivity)

    def add_noise(self, value: Union[int, float]) -> float:
        return self._mechanism.add_noise(1.0 * value)

    @property
    def noise_kind(self) -> pipeline_dp.NoiseKind:
        return pipeline_dp.NoiseKind.GAUSSIAN

    @property
    def noise_parameter(self) -> float:
        return self._mechanism.std

    @property
    def std(self) -> float:
        return self._mechanism.std

    @property
    def sensitivity(self) -> float:
        return self._mechanism.l2_sensitivity

    def describe(self) -> str:
        return (f"Gaussian mechanism:  parameter={self.noise_parameter}  eps="
                f"{self._mechanism.epsilon}  delta={self._mechanism.delta}  "
                f"l2_sensitivity={self.sensitivity}")


class MeanMechanism:
    """Computes DP mean.

    It computes DP mean as a ratio of DP sum and DP count. For improving
    utility the normalization to mid = (min_value + max_value)/2 is performed.
    It works in the following way:
    1. normalized_sum = sum(x_i-mid), where mid = (min_value+max_value)/2.
    2. dp_normalized_sum, dp_count are computed by adding Laplace or Gaussian
      noise.
    3. dp_mean = dp_normalized_sum/dp_count + mid.

    This normalization has benefits that normalized_sum has sensitivity
    (max_value-min_value)/2 which is smaller that
     sum_sensitivity = max(|min_value|, |max_value|).
    """

    def __init__(self, range_middle: float, count_mechanism: AdditiveMechanism,
                 sum_mechanism: AdditiveMechanism):
        self._range_middle = range_middle
        self._count_mechanism = count_mechanism
        self._sum_mechanism = sum_mechanism

    def compute_mean(self, count: int, normalized_sum: float):
        dp_count = self._count_mechanism.add_noise(count)
        denominator = max(1.0, dp_count)  # to avoid division on a small number.
        dp_normalized_sum = self._sum_mechanism.add_noise(normalized_sum)
        dp_mean = self._range_middle + dp_normalized_sum / denominator
        dp_sum = dp_mean * dp_count
        return dp_count, dp_sum, dp_mean

    def describe(self) -> str:
        return (
            f"    a. Computed 'normalized_sum' = sum of (value - {self._range_middle})\n"
            f"    b. Applied to 'count' {self._count_mechanism.describe()}\n"
            f"    c. Applied to 'normalized_sum' {self._sum_mechanism.describe()}"
        )


@dataclass
class Sensitivities:
    """Contains sensitivities of the additive DP mechanism."""
    l0: Optional[int] = None
    linf: Optional[float] = None
    l1: Optional[float] = None
    l2: Optional[float] = None

    def __post_init__(self):

        def check_is_positive(num: Any, name: str) -> bool:
            if num is not None and num <= 0:
                raise ValueError(f"{name} must be positive, but {num} given.")

        check_is_positive(self.l0, "L0")
        check_is_positive(self.linf, "Linf")
        check_is_positive(self.l1, "L1")
        check_is_positive(self.l2, "L2")

        if (self.l0 is None) != (self.linf is None):
            raise ValueError("l0 and linf sensitivities must be either both set"
                             " or both unset.")

        if self.l0 is not None and self.linf is not None:
            # Compute L1 sensitivity if not given, otherwise check that it is
            # correct.
            l1 = compute_l1_sensitivity(self.l0, self.linf)
            if self.l1 is None:
                self.l1 = l1
            else:
                if abs(l1 - self.l1) > 1e-12:
                    raise ValueError(f"L1={self.l1} != L0*Linf={l1}")

            # Compute L2 sensitivity if not given, otherwise check that it is
            # correct.
            l2 = compute_l2_sensitivity(self.l0, self.linf)
            if self.l2 is None:
                self.l2 = l2
            else:
                if abs(l2 - self.l2) > 1e-12:
                    raise ValueError(f"L2={self.l2} != sqrt(L0)*Linf={l2}")


@dataclass
class AdditiveMechanismSpec:
    """Contains the budget and noise_kind."""
    epsilon: float
    delta: float
    noise_kind: pipeline_dp.NoiseKind


def to_additive_mechanism_spec(
        spec: pipeline_dp.budget_accounting.MechanismSpec
) -> AdditiveMechanismSpec:
    noise_kind = spec.mechanism_type.to_noise_kind()
    return AdditiveMechanismSpec(spec.eps, spec.delta, noise_kind)


def create_additive_mechanism(
        spec: AdditiveMechanismSpec,
        sensitivities: Sensitivities) -> AdditiveMechanism:
    """Creates AdditiveMechanism from a mechanism spec and sensitivities."""
    if spec.noise_kind == pipeline_dp.NoiseKind.LAPLACE:
        if sensitivities.l1 is None:
            raise ValueError("L1 or (L0 and Linf) sensitivities must be set for"
                             " Laplace mechanism.")
        return LaplaceMechanism(spec.epsilon, sensitivities.l1)

    if spec.noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
        if sensitivities.l2 is None:
            raise ValueError("L2 or (L0 and Linf) sensitivities must be set for"
                             " Gaussian mechanism.")
        return GaussianMechanism(spec.epsilon, spec.delta, sensitivities.l2)

    assert False, f"{spec.noise_kind} not supported."


def create_mean_mechanism(
        range_middle: float, count_spec: AdditiveMechanismSpec,
        count_sensitivities: Sensitivities,
        normalized_sum_spec: AdditiveMechanismSpec,
        normalized_sum_sensitivities: Sensitivities) -> MeanMechanism:
    """Creates MeanMechanism from a mechanism specs and sensitivities."""
    count_mechanism = create_additive_mechanism(count_spec, count_sensitivities)
    sum_mechanism = create_additive_mechanism(normalized_sum_spec,
                                              normalized_sum_sensitivities)
    return MeanMechanism(range_middle, count_mechanism, sum_mechanism)


class ExponentialMechanism:
    """Exponential mechanism that can be used to choose a parameter
    from a set of possible parameters in a differentially private way.

    All computations are in memory, meaning that the set of possible parameters
    should fit in memory.

    https://en.wikipedia.org/wiki/Exponential_mechanism"""

    class ScoringFunction(abc.ABC):
        """Represents scoring function used in exponential mechanism."""

        @abc.abstractmethod
        def score(self, k) -> float:
            """Calculates score for the given parameter.

            The higher the score the greater the probability that
            this parameter will be chosen."""

        @property
        @abc.abstractmethod
        def global_sensitivity(self) -> float:
            """Global sensitivity of the scoring function."""

        @property
        @abc.abstractmethod
        def is_monotonic(self) -> bool:
            """Whether score(k) is monotonic.

            score(D, k), where D is the dataset, is monotonic
            if for any neighboring datasets D and D',
            either score(D, k) >= score(D', k) for any k or
            score(D, k) <= score(D', k) for any k."""

    def __init__(self, scoring_function: ScoringFunction) -> None:
        self._scoring_function = scoring_function

    def apply(self, eps: float, inputs_to_score_col: typing.List[Any]) -> Any:
        """Applies exponential mechanism.

        I.e. chooses a parameter from the list of possible parameters in a
        differentially private way."""

        probs = self._calculate_probabilities(eps, inputs_to_score_col)
        return np.random.default_rng().choice(inputs_to_score_col, p=probs)

    def _calculate_probabilities(self, eps: float,
                                 inputs_to_score_col: typing.List[Any]):
        scores = np.array(
            list(map(self._scoring_function.score, inputs_to_score_col)))
        denominator = self._scoring_function.global_sensitivity
        if not self._scoring_function.is_monotonic:
            denominator *= 2
        weights = np.exp(scores * eps / denominator)
        return weights / weights.sum()


def compute_sensitivities_for_count(
        params: pipeline_dp.AggregateParams) -> Sensitivities:
    if params.max_contributions is not None:
        return Sensitivities(l1=params.max_contributions,
                             l2=params.max_contributions)
    return Sensitivities(l0=params.max_partitions_contributed,
                         linf=params.max_contributions_per_partition)


def compute_sensitivities_for_privacy_id_count(
        params: pipeline_dp.AggregateParams) -> Sensitivities:
    return Sensitivities(l1=params.max_contributions,
                         l2=math.sqrt(params.max_contributions))
    return Sensitivities(l0=params.max_partitions_contributed, linf=1)


def compute_sensitivities_for_sum(
        params: pipeline_dp.AggregateParams) -> Sensitivities:
    l0_sensitivity = params.max_partitions_contributed
    max_abs_values = lambda x, y: max(abs(x), abs(y))
    if params.bounds_per_contribution_are_set:
        max_abs_val = max_abs_values(params.min_value, params.max_value)
        if params.max_contributions:
            l1_l2_sensitivity = max_abs_val * params.max_contributions
            return Sensitivities(l1=l1_l2_sensitivity, l2=l1_l2_sensitivity)
        linf_sensitivity = max_abs_val * params.max_contributions_per_partition
    else:
        linf_sensitivity = max_abs_values(params.min_sum_per_partition,
                                          params.max_sum_per_partition)
    return Sensitivities(l0=l0_sensitivity, linf=linf_sensitivity)


def compute_sensitivities_for_normalized_sum(
        params: pipeline_dp.AggregateParams) -> Sensitivities:
    max_abs_value = (params.max_value - params.min_value) / 2
    if params.max_contributions:
        l1_l2_sensitivity = max_abs_value * params.max_contributions
        return Sensitivities(l1=l1_l2_sensitivity, l2=l1_l2_sensitivity)
    l0_sensitivity = params.max_partitions_contributed
    linf_sensitivity = max_abs_value * params.max_contributions_per_partition

    return Sensitivities(l0=l0_sensitivity, linf=linf_sensitivity)
