"""Privacy budget accounting for DP pipelines."""

import logging
import math
from dataclasses import dataclass
from pipeline_dp.aggregate_params import MechanismType
from dp_accounting import privacy_loss_distribution as pldlib
from dp_accounting import common


@dataclass
class Budget:
    """Manages the budget allocated for an operation.

    The values for eps and delta are computed when the method compute_budgets
    of the corresponding BudgetAccount is called.
    """
    _eps: float = None
    _delta: float = None

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

    def set_eps_delta(self, eps, delta):
        self._eps = eps
        self._delta = delta


@dataclass
class RequestedBudget:
    """Manages the budget requested for an operation."""
    budget: Budget
    weight: float
    use_eps: bool
    use_delta: bool


class BudgetAccountant:
    """Manages the privacy budget."""

    def __init__(self, epsilon: float, delta: float):
        """Constructs a BudgetAccountant.

        Args:
            epsilon, delta: Parameters of (epsilon, delta)-differential privacy.
        """

        _validate_epsilon_delta(epsilon, delta)

        self._eps = epsilon
        self._delta = delta
        self._requested_budgets = []

    def request_budget(self, weight: float, *, use_eps: bool,
                       use_delta: bool) -> Budget:
        """Requests a budget.

        Args:
            weight: The weight used to compute epsilon and delta for the budget.
            use_eps: False when the operation doesn't need epsilon.
            use_delta: False when the operation doesn't need delta.

        Returns:
            A "lazy" budget object that doesn't contain epsilon/delta until the
            method compute_budgets is called.
        """
        budget = Budget()
        requested_budget = RequestedBudget(budget, weight, use_eps, use_delta)
        self._requested_budgets.append(requested_budget)
        return budget

    def compute_budgets(self):
        """Updates all previously requested Budget objects with corresponding budget values."""
        if not self._requested_budgets:
            logging.warning("No budgets were requested.")
            return

        total_weight_eps = total_weight_delta = 0
        for requested_budget in self._requested_budgets:
            total_weight_eps += requested_budget.use_eps * requested_budget.weight
            total_weight_delta += requested_budget.use_delta * requested_budget.weight

        for requested_budget in self._requested_budgets:
            eps = delta = 0
            if total_weight_eps:
                numerator = requested_budget.use_eps * self._eps * requested_budget.weight
                eps = numerator / total_weight_eps
            if total_weight_delta:
                numerator = requested_budget.use_delta * self._delta * requested_budget.weight
                delta = numerator / total_weight_delta
            requested_budget.budget.set_eps_delta(eps, delta)


@dataclass
class MechanismSpec:
    """Specifies the parameters for a mechanism.

    MechanismType defines the type of noise distribution.
    noise is the minimized noise standard deviation.
    """
    mechanism_type: MechanismType
    _noise_standard_deviation: float = None

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


@dataclass
class MechanismSpecInternal:
    """Stores sensitivity and weight not exposed in MechanismSpec."""
    sensitivity: float
    weight: float
    mechanism_spec: MechanismSpec


class PLDBudgetAccountant:
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
                       mechanism_type: MechanismType,
                       sensitivity: float = 1,
                       weight: float = 1) -> MechanismSpec:
        """Request a budget.

        Constructs a mechanism spec based on the parameters.
        Adds the mechanism to the pipeline for future calculation.

        Args:
            mechanism_type: The type of noise distribution for the mechanism.
            sensitivity: The sensitivity for the mechanism.
            weight: The weight for the mechanism.

        Returns:
            A "lazy" mechanism spec object that doesn't contain the noise
            standard deviation until compute_budgets is called.
        """
        if mechanism_type == MechanismType.GAUSSIAN and self._total_delta == 0:
            raise AssertionError(
                "The Gaussian mechanism requires that the pipeline delta is greater than 0"
            )
        mechanism_spec = MechanismSpec(mechanism_type=mechanism_type)
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
            if mechanism_spec_internal.mechanism_spec.mechanism_type == MechanismType.LAPLACE:
                # The Laplace distribution parameter = std/sqrt(2).
                pld = pldlib.PrivacyLossDistribution.from_laplace_mechanism(
                    mechanism_spec_internal.sensitivity *
                    noise_standard_deviation / math.sqrt(2) /
                    mechanism_spec_internal.weight,
                    value_discretization_interval=self._pld_discretization)
            elif mechanism_spec_internal.mechanism_spec.mechanism_type == MechanismType.GAUSSIAN:
                pld = pldlib.PrivacyLossDistribution.from_gaussian_mechanism(
                    mechanism_spec_internal.sensitivity *
                    noise_standard_deviation / mechanism_spec_internal.weight,
                    value_discretization_interval=self._pld_discretization)
            elif mechanism_spec_internal.mechanism_spec.mechanism_type == MechanismType.GENERIC:
                # It is required to convert between the noise_standard_deviation of a Laplace or Gaussian mechanism
                # and the (epsilon, delta) Generic mechanism because the calibration is defined by one parameter.
                # There are multiple ways to do this; here it is assumed that (epsilon, delta) specifies the Laplace
                # mechanism and epsilon is computed based on this. The delta is computed to be proportional to epsilon.
                epsilon_0 = math.sqrt(2) / noise_standard_deviation
                delta_0 = epsilon_0 / self._total_epsilon * self._total_delta
                pld = pldlib.PrivacyLossDistribution.from_privacy_parameters(
                    common.DifferentialPrivacyParameters(epsilon_0, delta_0),
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
