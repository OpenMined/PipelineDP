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
"""Privacy budget accounting for DP pipelines."""

import abc
import collections
import logging
import math
from typing import Optional

from dataclasses import dataclass
import pipeline_dp.aggregate_params as agg_params
from pipeline_dp import input_validators

try:
    from dp_accounting import privacy_loss_distribution as pldlib
    from dp_accounting import common
except:
    # dp_accounting library is needed only for PLDBudgetAccountant which is
    # currently in experimental mode.
    pass


@dataclass
class MechanismSpec:
    """Specifies the parameters for a DP mechanism.

    MechanismType defines the kind of noise distribution.
    _noise_standard_deviation is the minimized noise standard deviation.
    (_eps, _delta) are parameters of (eps, delta)-differential privacy
    """
    mechanism_type: agg_params.MechanismType
    _noise_standard_deviation: float = None
    _eps: float = None
    _delta: float = None
    _count: int = 1

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

    @property
    def count(self):
        """The number of times the mechanism is going to be applied"""
        return self._count

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
        return self.mechanism_type != agg_params.MechanismType.LAPLACE


@dataclass
class MechanismSpecInternal:
    """Stores sensitivity and weight not exposed in MechanismSpec."""
    sensitivity: float
    weight: float
    mechanism_spec: MechanismSpec


Budget = collections.namedtuple("Budget", ["epsilon", "delta"])


class BudgetAccountant(abc.ABC):
    """Base class for budget accountants."""

    def __init__(self, total_epsilon: float, total_delta: float,
                 num_aggregations: Optional[int],
                 aggregation_weights: Optional[list]):

        input_validators.validate_epsilon_delta(total_epsilon, total_delta,
                                                "BudgetAccountant")
        self._total_epsilon = total_epsilon
        self._total_delta = total_delta

        self._scopes_stack = []
        self._mechanisms = []
        self._finalized = False
        if num_aggregations is not None and aggregation_weights is not None:
            raise ValueError(
                "'num_aggregations' and 'aggregation_weights' can not be set "
                "simultaneously.\nIf you wish all aggregations in the pipeline "
                "to have equal budgets, specify the total number of aggregations"
                "with 'n_aggregations'.\nIf you wish to have different budgets "
                "for different aggregations, specify them with 'aggregation_weights'"
            )
        if num_aggregations is not None and num_aggregations <= 0:
            raise ValueError(
                f"'num_aggregations'={num_aggregations}, but it has to be positive."
            )
        self._expected_num_aggregations = num_aggregations
        self._expected_aggregation_weights = aggregation_weights
        self._reqested_aggregations = 0
        self._actual_aggregation_weights = []

    @abc.abstractmethod
    def request_budget(
            self,
            mechanism_type: agg_params.MechanismType,
            sensitivity: float = 1,
            weight: float = 1,
            count: int = 1,
            noise_standard_deviation: Optional[float] = None) -> MechanismSpec:
        pass

    @abc.abstractmethod
    def compute_budgets(self):
        pass

    def scope(self, weight: float):
        """Defines a scope for DP operations that should consume no more than "weight" proportion of the budget
        of the parent scope.

        The accountant will automatically scale the budgets of all sub-operations accordingly.

        Example usage:
          with accountant.scope(weight = 0.5):
             ... some code that consumes DP budget ...

        Args:
            weight: budget weight of all operations made within this scope as compared to.

        Returns:
            the scope that should be used in a "with" block enclosing the operations consuming the budget.
        """
        return BudgetAccountantScope(self, weight)

    def _compute_budget_for_aggregation(self, weight: float) -> Budget:
        """Computes budget per aggregation.

        It splits the budget using the naive composition.

        Warning: This function changes the 'self' internal state. It can be
        called only from the API function of DPEngine, like aggregate() or
        select_partitions().

        Args:
            weight: the budget weight of the aggregation.

        Returns:
            the budget.
        """
        self._actual_aggregation_weights.append(weight)
        if self._expected_num_aggregations:
            return Budget(self._total_epsilon / self._expected_num_aggregations,
                          self._total_delta / self._expected_num_aggregations)
        if self._expected_aggregation_weights:
            budget_ratio = weight / sum(self._expected_aggregation_weights)
            return Budget(self._total_epsilon * budget_ratio,
                          self._total_delta * budget_ratio)
        # No expectations on aggregations, no way to compute budget.
        return None

    def _check_aggregation_restrictions(self):
        if self._expected_num_aggregations:
            actual_num_aggregations = len(self._actual_aggregation_weights)
            if actual_num_aggregations != self._expected_num_aggregations:
                raise ValueError(
                    f"'num_aggregations'({self._expected_num_aggregations}) in "
                    f"the constructor of BudgetAccountant is different from the"
                    f" actual number of aggregations in the pipeline"
                    f"({actual_num_aggregations}). If 'n_aggregations' is "
                    f"specified, you must have that many aggregations in the "
                    f"pipeline.")
            weights = self._actual_aggregation_weights
            if not all([w == 1 for w in weights]):
                raise ValueError(
                    f"Aggregation weights = {weights}. If 'num_aggregations' is"
                    f" set in the constructor of BudgetAccountant, all "
                    f"aggregation weights have to be 1. If you'd like to have "
                    f"different weights use 'aggregation_weights'.")
        if self._expected_aggregation_weights:
            actual_weights = self._actual_aggregation_weights
            expected_weights = self._expected_aggregation_weights
            if len(actual_weights) != len(expected_weights):
                raise ValueError(
                    f"Length of 'aggregation_weights' in the constructor of "
                    f"BudgetAccountant is {len(expected_weights)} != "
                    f"{len(actual_weights)} the actual number of aggregations.")
            if not all(
                [w1 == w2 for w1, w2 in zip(actual_weights, expected_weights)]):
                raise ValueError(
                    f"'aggregation_weights' in the constructor of is "
                    f"({expected_weights}) is different from actual aggregation"
                    f" weights ({actual_weights}).If 'aggregation_weights' is "
                    f"specified, they must be the same.")

    def _register_mechanism(self, mechanism: MechanismSpecInternal):
        """Registers this mechanism for the future normalisation."""

        # Register in the global list of mechanisms
        self._mechanisms.append(mechanism)

        # Register in all of the current scopes
        for scope in self._scopes_stack:
            scope.mechanisms.append(mechanism)

        return mechanism

    def _enter_scope(self, scope):
        self._scopes_stack.append(scope)

    def _exit_scope(self):
        self._scopes_stack.pop()

    def _finalize(self):
        if self._finalized:
            raise Exception("compute_budgets can not be called twice.")
        self._finalized = True


@dataclass
class BudgetAccountantScope:

    def __init__(self, accountant, weight):
        self.weight = weight
        self.accountant = accountant
        self.mechanisms = []

    def __enter__(self):
        self.accountant._enter_scope(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accountant._exit_scope()
        self._normalise_mechanism_weights()

    def _normalise_mechanism_weights(self):
        """Normalise all mechanism weights so that they sum up to the weight of the current scope."""

        if not self.mechanisms:
            return

        total_weight = sum([m.weight for m in self.mechanisms])
        normalisation_factor = self.weight / total_weight
        for mechanism in self.mechanisms:
            mechanism.weight *= normalisation_factor


class NaiveBudgetAccountant(BudgetAccountant):
    """Manages the privacy budget."""

    def __init__(self,
                 total_epsilon: float,
                 total_delta: float,
                 num_aggregations: Optional[int] = None,
                 aggregation_weights: Optional[list] = None):
        """Constructs a NaiveBudgetAccountant.

        Args:
            total_epsilon: epsilon for the entire pipeline.
            total_delta: delta for the entire pipeline.
            num_aggregations: number of DP aggregations in the pipeline for
             which  'self' manages the budget. All aggregations should have
             'budget_weight' = 1. When specified, all aggregations will have
             equal budget. It is useful to ensure that the pipeline has fixed
             number of DP aggregations.
            aggregation_weights: 'budget_weight' of aggregations for which
             'self' manages the budget. It is useful to ensure that the pipeline
              has a fixed number of DP aggregations with fixed weights.

        If num_aggregations and aggregation_weights are not set, there are no
        restrictions on the number of aggregations nor their budget weights.

        Raises:
            A ValueError if either argument is out of range.
            A ValueError if num_aggregations and aggregation_weights are both set.
        """
        super().__init__(total_epsilon, total_delta, num_aggregations,
                         aggregation_weights)

    def request_budget(
            self,
            mechanism_type: agg_params.MechanismType,
            sensitivity: float = 1,
            weight: float = 1,
            count: int = 1,
            noise_standard_deviation: Optional[float] = None) -> MechanismSpec:
        """Requests a budget.

        Constructs a mechanism spec based on the parameters.
        Keeps the mechanism spec for future calculations.

        Args:
            mechanism_type: The type of noise distribution for the mechanism.
            sensitivity: The sensitivity for the mechanism.
            weight: The weight for the mechanism.
            count: The number of times the mechanism will be applied.
            noise_standard_deviation: The standard deviation for the mechanism.

        Returns:
            A "lazy" mechanism spec object that doesn't contain the noise
            standard deviation until compute_budgets is called.
        """
        if self._finalized:
            raise Exception(
                "request_budget() is called after compute_budgets(). "
                "Please ensure that compute_budgets() is called after DP "
                "aggregations.")

        if noise_standard_deviation is not None:
            raise NotImplementedError(
                "Count and noise standard deviation have not been implemented yet."
            )
        if mechanism_type == agg_params.MechanismType.GAUSSIAN and self._total_delta == 0:
            raise ValueError(
                "The Gaussian mechanism requires that the pipeline delta is greater than 0"
            )
        mechanism_spec = MechanismSpec(mechanism_type=mechanism_type,
                                       _count=count)
        mechanism_spec_internal = MechanismSpecInternal(
            mechanism_spec=mechanism_spec,
            sensitivity=sensitivity,
            weight=weight)

        self._register_mechanism(mechanism_spec_internal)
        return mechanism_spec

    def compute_budgets(self):
        """Updates all previously requested MechanismSpec objects with corresponding budget values."""
        self._check_aggregation_restrictions()
        self._finalize()

        if not self._mechanisms:
            logging.warning("No budgets were requested.")
            return

        if self._scopes_stack:
            raise Exception(
                "Cannot call compute_budgets from within a budget scope.")

        total_weight_eps = total_weight_delta = 0
        for mechanism in self._mechanisms:
            total_weight_eps += mechanism.weight * mechanism.mechanism_spec.count
            if mechanism.mechanism_spec.use_delta():
                total_weight_delta += mechanism.weight * mechanism.mechanism_spec.count

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

    This class is experimental. It is not yet compatible with DPEngine.
    """

    def __init__(self,
                 total_epsilon: float,
                 total_delta: float,
                 pld_discretization: float = 1e-4,
                 num_aggregations: Optional[int] = None,
                 aggregation_weights: Optional[list] = None):
        """Constructs a PLDBudgetAccountant.

        Args:
            total_epsilon: epsilon for the entire pipeline.
            total_delta: delta for the entire pipeline.
            pld_discretization: `value_discretization_interval` in PLD library.
                Smaller interval results in better accuracy, but increases running time.
              num_aggregations: number of DP aggregations in the pipeline for
             which  'self' manages the budget. All aggregations should have
             'budget_weight' = 1. When specified all aggregations will have
             equal budget. It is useful to ensure that the pipeline has fixed
             number of DP aggregations.
            aggregation_weights: 'budget_weight' of aggregations for which
             'self' manages the budget. It is useful to ensure that the pipeline
              has a fixed number of DP aggregations with fixed weights.

        If num_aggregations and aggregation_weights are not set, there are no
        restrictions on the number of aggregations nor their budget weights.


        Raises:
            ValueError: Arguments are missing or out of range.
        """

        super().__init__(total_epsilon, total_delta, num_aggregations,
                         aggregation_weights)

        self.minimum_noise_std = None
        self._pld_discretization = pld_discretization

    def request_budget(
            self,
            mechanism_type: agg_params.MechanismType,
            sensitivity: float = 1,
            weight: float = 1,
            count: int = 1,
            noise_standard_deviation: Optional[float] = None) -> MechanismSpec:
        """Request a budget.

        Constructs a mechanism spec based on the parameters.
        Adds the mechanism to the pipeline for future calculation.

        Args:
            mechanism_type: The type of noise distribution for the mechanism.
            sensitivity: The sensitivity for the mechanism.
            weight: The weight for the mechanism.
            count: The number of times the mechanism will be applied.
            noise_standard_deviation: The standard deviation for the mechanism.


        Returns:
            A "lazy" mechanism spec object that doesn't contain the noise
            standard deviation until compute_budgets is called.
        """
        if self._finalized:
            raise Exception(
                "request_budget() is called after compute_budgets(). "
                "Please ensure that compute_budgets() is called after DP "
                "aggregations.")

        if count != 1 or noise_standard_deviation is not None:
            raise NotImplementedError(
                "Count and noise standard deviation have not been implemented yet."
            )
        if mechanism_type == agg_params.MechanismType.GAUSSIAN and self._total_delta == 0:
            raise AssertionError(
                "The Gaussian mechanism requires that the pipeline delta is greater than 0"
            )
        mechanism_spec = MechanismSpec(mechanism_type=mechanism_type)
        mechanism_spec_internal = MechanismSpecInternal(
            mechanism_spec=mechanism_spec,
            sensitivity=sensitivity,
            weight=weight)
        self._register_mechanism(mechanism_spec_internal)
        return mechanism_spec

    def compute_budgets(self):
        """Computes the budget for the pipeline.

        Composes the mechanisms and adjusts the amount of
        noise based on given epsilon. Sets the noise for the
        entire pipeline.
        """
        self._check_aggregation_restrictions()
        self._finalize()

        if not self._mechanisms:
            logging.warning("No budgets were requested.")
            return

        if self._scopes_stack:
            raise Exception(
                "Cannot call compute_budgets from within a budget scope.")

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
            if mechanism.mechanism_spec.mechanism_type == agg_params.MechanismType.GENERIC:
                epsilon_0 = math.sqrt(2) / mechanism_noise_std
                delta_0 = epsilon_0 / self._total_epsilon * self._total_delta
                mechanism.mechanism_spec.set_eps_delta(epsilon_0, delta_0)

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
            self, noise_standard_deviation: float
    ) -> 'pldlib.PrivacyLossDistribution':
        """Uses the Privacy Loss Distribution library to compose distributions.

        Args:
            noise_standard_deviation: The noise of the distributions to construct.

        Returns:
            A PrivacyLossDistribution object for the pipeline.
        """
        composed, pld = None, None

        for mechanism_spec_internal in self._mechanisms:
            if mechanism_spec_internal.mechanism_spec.mechanism_type == agg_params.MechanismType.LAPLACE:
                # The Laplace distribution parameter = std/sqrt(2).
                pld = pldlib.PrivacyLossDistribution.from_laplace_mechanism(
                    mechanism_spec_internal.sensitivity *
                    noise_standard_deviation / math.sqrt(2) /
                    mechanism_spec_internal.weight,
                    value_discretization_interval=self._pld_discretization)
            elif mechanism_spec_internal.mechanism_spec.mechanism_type == agg_params.MechanismType.GAUSSIAN:
                pld = pldlib.PrivacyLossDistribution.from_gaussian_mechanism(
                    mechanism_spec_internal.sensitivity *
                    noise_standard_deviation / mechanism_spec_internal.weight,
                    value_discretization_interval=self._pld_discretization)
            elif mechanism_spec_internal.mechanism_spec.mechanism_type == agg_params.MechanismType.GENERIC:
                # It is required to convert between the noise_standard_deviation of a Laplace or Gaussian mechanism
                # and the (epsilon, delta) Generic mechanism because the calibration is defined by one parameter.
                # There are multiple ways to do this; here it is assumed that (epsilon, delta) specifies the Laplace
                # mechanism and epsilon is computed based on this. The delta is computed to be proportional to epsilon.
                epsilon_0_interim = math.sqrt(2) / noise_standard_deviation
                delta_0_interim = epsilon_0_interim / self._total_epsilon * self._total_delta
                pld = pldlib.PrivacyLossDistribution.from_privacy_parameters(
                    common.DifferentialPrivacyParameters(
                        epsilon_0_interim, delta_0_interim),
                    value_discretization_interval=self._pld_discretization)

            composed = pld if composed is None else composed.compose(pld)

        return composed
