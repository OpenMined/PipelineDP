"""Privacy budget accounting for DP pipelines."""


class BudgetAccountant:
  """Manages privacy budget."""

  def __init__(self, epsilon, delta):
    """Construct a BudgetAccountant

    Args:
      epsilon, delta: parameters of (epsilon, delta)-differential privacy.
    """
    if epsilon <= 0:
      raise ValueError(f"Epsilon must be positive, not {epsilon}")
    if delta < 0:
      raise ValueError(f"Delta must non-negative, not {delta}")
    self._eps = epsilon
    self._delta = delta

  # TODO: implement BudgetAccountant functionality.
