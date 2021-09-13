"""Privacy budget accounting for DP pipelines."""

import abc
import logging
import math
from dataclasses import dataclass
from typing import Optional
from pipeline_dp.aggregate_params import NoiseKind
from dp_accounting import privacy_loss_distribution as pldlib


@dataclass
class MechanismSpec:
    """Specifies the parameters for a DP mechanism.

    NoiseKind defines the kind of noise distribution.
    _noise_standard_deviation is the minimized noise standard deviation.
    (_eps, _delta) are parameters of (eps, delta)-differential privacy
    """
    noise_kind: NoiseKind
    _noise_standard_deviation: float = None
    _eps: float = None
    _delta: float = None

    @property
    def noise_standard_deviation(self):
        """Noise value for the mechanism.

        Raises:
            AssertionError: The noise value is not calculated yet.
        """
        if self._noise_standard_deviation is None:
            raise AssertionError(
                "Noise standard deviation is not calculated yet.")
        return self._noise_standard_deviation

    @property
    def eps(self):
        """Parameter of (eps, delta)-differential privacy.
               Raises:
                   AssertionError: The privacy budget is not calculated yet.
       """
        if self._eps is None:
            raise AssertionError("Privacy budget is not calculated yet.")
        return self._eps

    @property
    def delta(self):
        """Parameter of (eps, delta)-differential privacy.
                Raises:
                    AssertionError: The privacy budget is not calculated yet.
        """
        if self._delta is None:
            raise AssertionError("Privacy budget is not calculated yet.")
        return self._delta

    def set_eps_delta(self, eps: float, delta: Optional[float]) -> None:
        """Set parameters for (eps, delta)-differential privacy.

        Raises:
            AssertionError: eps must not be None.
        """
        if eps is None:
            raise AssertionError("eps must not be None.")
        self._eps = eps
        self._delta = delta
        return

    def use_delta(self) -> bool:
        return self.noise_kind == NoiseKind.GAUSSIAN


@dataclass
class MechanismSpecInternal:
    """Stores sensitivity and weight not exposed in MechanismSpec."""
    sensitivity: float
    weight: float
    mechanism_spec: MechanismSpec


class BudgetAccountant(abc.ABC):
    """Base class for budget accountants."""

    @abc.abstractmethod
    def request_budget(self,
                       noise_kind: NoiseKind,
                       sensitivity: float = 1,
                       weight: float = 1) -> MechanismSpec:
        pass

    @abc.abstractmethod
    def compute_budgets(self):
        pass


class NaiveBudgetAccountant(BudgetAccountant):
    """Manages the privacy budget."""

    def __init__(self, total_epsilon: float, total_delta: float):
        """Constructs a NaiveBudgetAccountant.

        Args:
            total_epsilon: epsilon for the entire pipeline.
            total_delta: delta for the entire pipeline.

        Raises:
            A ValueError if either argument is out of range.
        """

        _validate_epsilon_delta(total_epsilon, total_delta)

        self._total_epsilon = total_epsilon
        self._total_delta = total_delta
        self._mechanisms = []

    def request_budget(self,
                       noise_kind: NoiseKind,
                       sensitivity: float = 1,
                       weight: float = 1) -> MechanismSpec:
        """Requests a budget.

        Constructs a mechanism spec based on the parameters.
        Keeps the mechanism spec for future calculations.

        Args:
            noise_kind: The kind of noise distribution for the mechanism.
            sensitivity: The sensitivity for the mechanism.
            weight: The weight for the mechanism.

        Returns:
            A "lazy" mechanism spec object that doesn't contain the noise
            standard deviation until compute_budgets is called.
        """
        if noise_kind == NoiseKind.GAUSSIAN and self._total_delta == 0:
            raise AssertionError(
                "The Gaussian mechanism requires that the pipeline delta is greater than 0"
            )
        mechanism_spec = MechanismSpec(noise_kind=noise_kind)
        mechanism_spec_internal = MechanismSpecInternal(
            mechanism_spec=mechanism_spec,
            sensitivity=sensitivity,
            weight=weight)
        self._mechanisms.append(mechanism_spec_internal)
        return mechanism_spec

    def compute_budgets(self):
        """Updates all previously requested MechanismSpec objects with corresponding budget values."""
        if not self._mechanisms:
            logging.warning("No budgets were requested.")
            return

        total_weight_eps = total_weight_delta = 0
        for mechanism in self._mechanisms:
            total_weight_eps += mechanism.weight
            if mechanism.mechanism_spec.use_delta():
                total_weight_delta += mechanism.weight

        for mechanism in self._mechanisms:
            eps = delta = 0
            if total_weight_eps:
                numerator = self._total_epsilon * mechanism.weight
                eps = numerator / total_weight_eps
            if mechanism.mechanism_spec.use_delta():
                if total_weight_delta:
                    numerator = self._total_delta * mechanism.weight
                    delta = numerator / total_weight_delta
            mechanism.mechanism_spec.set_eps_delta(eps, delta)


class PLDBudgetAccountant(BudgetAccountant):
    """Manages the privacy budget for privacy loss distributions.

    It manages the privacy budget for the pipeline using the
    Privacy Loss Distribution (PLD) implementation from Google's
    dp_accounting library.
    """

    def __init__(self,
                 total_epsilon: float,
                 total_delta: float,
                 pld_discretization: float = 1e-4):
        """Constructs a PLDBudgetAccountant.

        Args:
            total_epsilon: epsilon for the entire pipeline.
            total_delta: delta for the entire pipeline.
            pld_discretization: `value_discretization_interval` in PLD library.
                Smaller interval results in better accuracy, but increases running time.

        Raises:
            ValueError: Arguments are missing or out of range.
        """

        _validate_epsilon_delta(total_epsilon, total_delta)

        self._total_epsilon = total_epsilon
        self._total_delta = total_delta
        self._mechanisms = []
        self.minimum_noise_std = None
        self._pld_discretization = pld_discretization

    def request_budget(self,
                       noise_kind: NoiseKind,
                       sensitivity: float = 1,
                       weight: float = 1) -> MechanismSpec:
        """Request a budget.

        Constructs a mechanism spec based on the parameters.
        Adds the mechanism to the pipeline for future calculation.

        Args:
            noise_kind: The kind of noise distribution for the mechanism.
            sensitivity: The sensitivity for the mechanism.
            weight: The weight for the mechanism.

        Returns:
            A "lazy" mechanism spec object that doesn't contain the noise
            standard deviation until compute_budgets is called.
        """
        if noise_kind == NoiseKind.GAUSSIAN and self._total_delta == 0:
            raise AssertionError(
                "The Gaussian mechanism requires that the pipeline delta is greater than 0"
            )
        mechanism_spec = MechanismSpec(noise_kind=noise_kind)
        mechanism_spec_internal = MechanismSpecInternal(
            mechanism_spec=mechanism_spec,
            sensitivity=sensitivity,
            weight=weight)
        self._mechanisms.append(mechanism_spec_internal)
        return mechanism_spec

    def compute_budgets(self):
        """Computes the budget for the pipeline.

        Composes the mechanisms and adjusts the amount of
        noise based on given epsilon. Sets the noise for the
        entire pipeline.
        """
        if not self._mechanisms:
            return
        if self._total_delta == 0:
            sum_weights = 0
            for mechanism in self._mechanisms:
                sum_weights += mechanism.weight
            minimum_noise_std = sum_weights / self._total_epsilon * math.sqrt(2)
        else:
            minimum_noise_std = self._find_minimum_noise_std()

        self.minimum_noise_std = minimum_noise_std
        for mechanism in self._mechanisms:
            mechanism_noise_std = mechanism.sensitivity * minimum_noise_std / mechanism.weight
            mechanism.mechanism_spec._noise_standard_deviation = mechanism_noise_std

    def _find_minimum_noise_std(self) -> float:
        """Finds the minimum noise which satisfies the total budget.

        Use binary search to find a minimum noise value that gives a
        new epsilon close to the given epsilon (within a threshold).
        By increasing the noise we can decrease the epsilon.

        Returns:
            The noise value adjusted for the given epsilon.
        """
        threshold = 1e-4
        maximum_noise_std = self._calculate_max_noise_std()
        low, high = 0, maximum_noise_std
        while low + threshold < high:
            mid = (high - low) / 2 + low
            pld = self._compose_distributions(mid)
            pld_epsilon = pld.get_epsilon_for_delta(self._total_delta)
            if pld_epsilon <= self._total_epsilon:
                high = mid
            elif pld_epsilon > self._total_epsilon:
                low = mid

        return high

    def _calculate_max_noise_std(self) -> float:
        """Calculates an upper bound for the noise to satisfy the budget."""
        max_noise_std = 1
        pld_epsilon = self._total_epsilon + 1
        while pld_epsilon > self._total_epsilon:
            max_noise_std *= 2
            pld = self._compose_distributions(max_noise_std)
            pld_epsilon = pld.get_epsilon_for_delta(self._total_delta)
        return max_noise_std

    def _compose_distributions(
            self,
            noise_standard_deviation: float) -> pldlib.PrivacyLossDistribution:
        """Uses the Privacy Loss Distribution library to compose distributions.

        Args:
            noise_standard_deviation: The noise of the distributions to construct.

        Returns:
            A PrivacyLossDistribution object for the pipeline.
        """
        composed, pld = None, None

        for mechanism_spec_internal in self._mechanisms:
            if mechanism_spec_internal.mechanism_spec.noise_kind == NoiseKind.LAPLACE:
                # The Laplace distribution parameter = std/sqrt(2).
                pld = pldlib.PrivacyLossDistribution.from_laplace_mechanism(
                    mechanism_spec_internal.sensitivity *
                    noise_standard_deviation / math.sqrt(2) /
                    mechanism_spec_internal.weight,
                    value_discretization_interval=self._pld_discretization)
            elif mechanism_spec_internal.mechanism_spec.noise_kind == NoiseKind.GAUSSIAN:
                pld = pldlib.PrivacyLossDistribution.from_gaussian_mechanism(
                    mechanism_spec_internal.sensitivity *
                    noise_standard_deviation / mechanism_spec_internal.weight,
                    value_discretization_interval=self._pld_discretization)

            composed = pld if composed is None else composed.compose(pld)

        return composed


def _validate_epsilon_delta(epsilon: float, delta: float):
    """Helper function to validate the epsilon and delta parameters.

    Args:
        epsilon: The epsilon value to validate.
        delta: The delta value to validate.

    Raises:
        A ValueError if either epsilon or delta are out of range.
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, not {epsilon}.")
    if delta < 0:
        raise ValueError(f"Delta must be non-negative, not {delta}.")
