"""Privacy budget accounting for DP pipelines."""


class Budget:
  """Manages the budget allocated for an operation."""

  def __init__(self, weight, use_eps, use_delta, eps=None, delta=None):
    """Constructs a Budget.

    The values for eps and delta will be computed only after the pipeline was
    constructed and the method compute_budgets was called.

    Args:
      weight: The weight is used to get more accurate results.
      use_eps: A boolean that is False when the operation doesn't need epsilon.
      use_delta: A boolean that is False when the operation doesn't need delta.
    """
    self._eps = eps
    self._delta = delta
    self.weight = weight
    self.use_eps = use_eps
    self.use_delta = use_delta

  @property
  def eps(self):
    """eps: Optional; Parameter of (eps, delta)-differential privacy."""
    if self._eps is None:
      raise AssertionError("Privacy budget is not calculated yet.")
    return self._eps

  @property
  def delta(self):
    """delta: Optional; Parameter of (eps, delta)-differential privacy."""
    if self._delta is None:
      raise AssertionError("Privacy budget is not calculated yet.")
    return self._delta

  def set_eps_delta(self, eps, delta):
    self._eps = eps
    self._delta = delta


class BudgetAccountant:
  """Manages the privacy budget."""

  def __init__(self, epsilon, delta):
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

  def request_budget(self, weight, use_eps, use_delta):
    """Requests a budget.

    Args:
      weight: The weight used to compute epsilon and delta for the budget.
      use_eps: A boolean that is False when the operation doesn't need epsilon.
      use_delta: A boolean that is False when the operation doesn't need delta.

    Returns:
      A "lazy" budget object that doesn't contain epsilon/delta until the
      method compute_budgets is called.
    """
    budget = Budget(weight=weight, use_eps=use_eps, use_delta=use_delta)
    self._requested_budgets.append(budget)
    return budget

  def compute_budgets(self):
    """Computes the budgets after constructing the pipeline."""
    if not self._requested_budgets:
      print("No budgets were requested.")
      return

    total_weight = sum([budget.weight for budget in self._requested_budgets])

    for budget in self._requested_budgets:
      eps = 0
      delta = 0
      if total_weight != 0:
        eps = budget.use_eps * self._eps * budget.weight / total_weight
        delta = budget.use_delta * self._delta * budget.weight / total_weight
      budget.set_eps_delta(eps, delta)
