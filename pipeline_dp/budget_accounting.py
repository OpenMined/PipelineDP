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
    from dp_accounting.pld import privacy_loss_distribution as pldlib
    from dp_accounting.pld import common
except:
    # dp_accounting library is needed only for PLDBudgetAccountant which is
    # currently in experimental mode.
    pass


def _check_pldlib_imported() -> bool:
    import sys
    return "dp_accounting.pld.privacy_loss_distribution" in sys.modules


MechanismType = agg_params.MechanismType


@dataclass
class MechanismSpec:
    """Specifies the parameters for a DP mechanism.

    MechanismType defines the kind of noise distribution.
    _noise_standard_deviation is the minimized noise standard deviation.
    (_eps, _delta) are parameters of (eps, delta)-differential privacy
    """
    mechanism_type: MechanismType
    _noise_standard_deviation: Optional[float] = None
    _eps: Optional[float] = None
    _delta: Optional[float] = None
    _count: Optional[int] = 1
    _thresholding_delta: Optional[float] = None

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

    def set_noise_standard_deviation(self, stddev: float) -> None:
        self._noise_standard_deviation = stddev

    def set_thresholding_delta(self, delta: float) -> None:
        self._thresholding_delta = delta

    @property
    def thresholding_delta(self) -> float:
        return self._thresholding_delta

    def use_delta(self) -> bool:
        return self.mechanism_type != MechanismType.LAPLACE

    @property
    def standard_deviation_is_set(self) -> bool:
        return self._noise_standard_deviation is not None


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
            mechanism_type: MechanismType,
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
            mechanism_type: MechanismType,
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
        if mechanism_type == MechanismType.GAUSSIAN and self._total_delta == 0:
            raise ValueError(
                "The Gaussian mechanism requires that the pipeline delta is greater than 0"
            )
        if mechanism_type.is_partition_selection and self._total_delta == 0:
            raise ValueError(
                "The private partition selections requires that the pipeline delta is greater than 0"
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

        if not _check_pldlib_imported():
            raise ImportError("dp_accounting library is not imported. It is"
                              "required for using PLD budget accounting. "
                              "Please install dp_accounting library or use"
                              "NaiveBudgetAccountant instead of "
                              "PLDBudgetAccountant")

        self.base_noise_std = None
        self._pld_discretization = pld_discretization

    def request_budget(
            self,
            mechanism_type: MechanismType,
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
        if mechanism_type == MechanismType.GAUSSIAN and self._total_delta == 0:
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
            base_noise_std = sum_weights / self._total_epsilon * math.sqrt(2)
        else:
            base_noise_std = self._find_minimum_base_noise_std()

        self.base_noise_std = base_noise_std
        thresholding_delta_per_mechanism = 0
        if self._count_thresholding_mechanisms() > 0:
            thresholding_delta_per_mechanism = self._get_thresholding_delta(
            ) / self._count_thresholding_mechanisms()

        for mechanism in self._mechanisms:
            mechanism_noise_std = mechanism.sensitivity * base_noise_std / mechanism.weight
            spec = mechanism.mechanism_spec
            if spec.mechanism_type in [
                    MechanismType.GENERIC, MechanismType.TRUNCATED_GEOMETRIC
            ]:
                epsilon_0 = math.sqrt(2) / mechanism_noise_std
                delta_0 = epsilon_0 / self._total_epsilon * self._total_delta
                mechanism.mechanism_spec.set_eps_delta(epsilon_0, delta_0)
            else:
                spec.set_noise_standard_deviation(mechanism_noise_std)
            if spec.mechanism_type.is_thresholding_mechanism:
                spec.set_thresholding_delta(thresholding_delta_per_mechanism)

    def _find_minimum_base_noise_std(self) -> float:
        """Finds the minimum noise which satisfies the total budget.

        Use binary search to find a minimum noise value that gives a
        new epsilon close to the given epsilon (within a threshold).
        By increasing the noise we can decrease the epsilon.

        Returns:
            The noise value adjusted for the given epsilon.
        """
        delta = self._total_delta - self._get_thresholding_delta()
        threshold = 1e-4
        maximum_noise_std = self._calculate_max_noise_std()
        low, high = 0, maximum_noise_std
        while low + threshold < high:
            mid = (high - low) / 2 + low
            pld = self._compose_distributions(mid)
            pld_epsilon = pld.get_epsilon_for_delta(delta)
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
            self, noise_stddev: float) -> 'pldlib.PrivacyLossDistribution':
        """Uses the Privacy Loss Distribution library to compose distributions.

        Args:
            noise_stddev: The base noise standard deviation of the distributions
              to construct.

        Returns:
            A PrivacyLossDistribution which corresponds to self._mechanisms and
            the base noise 'noise_stddev'.
        """
        composed, pld = None, None

        for mechanism in self._mechanisms:
            mechanism_type = mechanism.mechanism_spec.mechanism_type
            if mechanism_type == MechanismType.LAPLACE:
                pld = self._create_pld_for_laplace(noise_stddev, mechanism)
            elif mechanism_type == MechanismType.LAPLACE_THRESHOLDING:
                # todo
                pld = self._create_pld_for_laplace(noise_stddev, mechanism)
            elif mechanism_type == MechanismType.GAUSSIAN:
                pld = self._create_pld_for_gaussian(noise_stddev, mechanism)
            elif mechanism_type == MechanismType.GAUSSIAN_THRESHOLDING:
                # todo
                pld = self._create_pld_for_gaussian(noise_stddev, mechanism)
            elif mechanism_type == MechanismType.GENERIC:
                pld = self._create_pld_for_generic(noise_stddev, mechanism)
            composed = pld if composed is None else composed.compose(pld)

        return composed

    def _count_thresholding_mechanisms(self):
        return len(self._thresholding_mechanisms())

    def _get_thresholding_delta(self) -> float:
        has_threshold_mechanisms = bool(self._thresholding_mechanisms())
        return 0.25 * self._total_delta if has_threshold_mechanisms else 0

    def _thresholding_mechanisms(self):
        result = []
        for m in self._mechanisms:
            if m.mechanism_spec.mechanism_type.is_thresholding_mechanism:
                result.append(m)
        return result

    def _create_pld_for_laplace(
            self, noise_stddev: float, mechanism: MechanismSpecInternal
    ) -> 'pldlib.PrivacyLossDistribution':
        # The Laplace distribution parameter = std/sqrt(2).
        laplace_b = mechanism.sensitivity * noise_stddev / math.sqrt(
            2) / mechanism.weight
        return pldlib.from_laplace_mechanism(
            laplace_b, value_discretization_interval=self._pld_discretization)

    def _create_pld_for_gaussian(
            self, noise_stddev: float, mechanism: MechanismSpecInternal
    ) -> 'pldlib.PrivacyLossDistribution':
        return pldlib.from_gaussian_mechanism(
            mechanism.sensitivity * noise_stddev / mechanism.weight,
            value_discretization_interval=self._pld_discretization)

    def _create_pld_for_generic(
            self, noise_stddev: float, mechanism: MechanismSpecInternal
    ) -> 'pldlib.PrivacyLossDistribution':
        # It is required to convert between the noise_standard_deviation
        # of a Laplace or Gaussian mechanism and the (epsilon, delta)
        # Generic mechanism because the calibration is defined by one
        # parameter. There are multiple ways to do this; here it is
        # assumed that (epsilon, delta) specifies the Laplace mechanism
        # and epsilon is computed based on this. The delta is computed
        # to be proportional to epsilon.
        epsilon_0_interim = math.sqrt(2) / noise_stddev
        delta_0_interim = epsilon_0_interim / self._total_epsilon * self._total_delta
        return pldlib.from_privacy_parameters(
            common.DifferentialPrivacyParameters(epsilon_0_interim,
                                                 delta_0_interim),
            value_discretization_interval=self._pld_discretization)
