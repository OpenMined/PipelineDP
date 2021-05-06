"""Privacy budget accounting for DP pipelines."""

import logging
from dataclasses import dataclass

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
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, not {epsilon}.")
        if delta < 0:
            raise ValueError(f"Delta must be non-negative, not {delta}.")

        self._eps = epsilon
        self._delta = delta
        self._requested_budgets = []

    def request_budget(self, weight: float, *, use_eps: bool, use_delta: bool) -> Budget:
        """Requests a budget.

        Args:
            weight: The weight used to compute epsilon and delta for the budget.
            use_eps: A boolean that is False when the operation doesn't need epsilon.
            use_delta: A boolean that is False when the operation doesn't need delta.

        Returns:
            A "lazy" budget object that doesn't contain epsilon/delta until the
            method compute_budgets is called.
        """
        budget = Budget()
        requested_budget = RequestedBudget(budget, weight, use_eps, use_delta)
        self._requested_budgets.append(requested_budget)
        return budget

    def compute_budgets(self):
        """All previously requested Budget objects are updated with corresponding budget values."""
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
                eps = requested_budget.use_eps * self._eps * requested_budget.weight / total_weight_eps
            if total_weight_delta:
                delta = requested_budget.use_delta * self._delta * requested_budget.weight / total_weight_delta
            requested_budget.budget.set_eps_delta(eps, delta)
