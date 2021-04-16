from dataclasses import dataclass

class Budget:
  def __init__(self, text:str, eps = None, delta = None, eps_scale = None, delta_scale = None):
    self._eps_scale = eps_scale
    self._delta_scale = delta_scale
    self._text = text
    self._eps = eps
    self._delta = delta

  @property
  def eps(self):
    assert self._eps is not None, "Privacy budget is not calculated yet."
    return self._eps

  @property
  def delta(self):
    assert self._delta is not None, "Privacy budget is not calculated yet."
    return self._delta

  def set_eps_delta(self, eps, delta):
    print(f"Budget for {self._text} is ({eps}, {delta})")
    self._eps = eps
    self._delta = delta

class BudgetAccountant:
  """Manages privacy budget

    request_budget() returns a "lazy" budget object, that doesn't contain
    eps/delta until compute_budgets() is called.
    compute_budget() should be called after finishing the pipeline contruction.
    Collecting all budget requests and then computing budget allows more
    effective budget usage.

    use_budget() returns already computed eps, delta.

    Now the basic budget composition is used.
    Eventually the dp_accounging library
     https://github.com/google/differential-privacy/tree/main/python/dp_accounting
    will be used. It contains the implementation of Privacy Loss Distribution (PLD)
    for the buget computing. PLD provides especially significant benefits in
    comparison with the basic composition in case of many DP aggregations.
  """

  def __init__(self, eps, delta):
    assert eps > 0 and delta >= 0, "DP params violation"
    self.eps = eps
    self.delta = delta
    self.left_eps = eps
    self.left_delta = delta
    self.requested_budgets = []

  def request_budget(self, eps_scale, delta_scale = 0, text = None):
    budget = Budget(text, eps_scale=eps_scale, delta_scale=delta_scale)
    self.requested_budgets.append(budget)
    return budget

  def use_budget(self, eps_ratio, delta_ratio = 0, text = None):
    assert self.left_eps > 0, "No more eps left" # todo provide more details
    eps_to_use = self.eps * eps_ratio
    delta_to_use = self.delta * delta_ratio
    assert not(self.left_delta == 0 and delta_to_use > 0)  , "No more delta left" # todo provide more details
    eps_to_use = min(eps_to_use, self.left_eps)
    self.left_eps -= eps_to_use
    delta_to_use = min(delta_to_use, self.left_delta)
    self.left_delta -= delta_to_use
    budget = Budget(text, eps=eps_to_use, delta=delta_to_use)
    return budget

  def compute_budgets(self):
    if not self.requested_budgets:
      print("No budgets were requested")
      return
    # basic composition, here is the place to introduce more advance methods in future.
    sum_eps = sum_delta = 0
    for budget in self.requested_budgets:
      sum_eps += budget._eps_scale
      sum_delta += budget._delta_scale
    for budget in self.requested_budgets:
      budget_eps = self.eps * budget._eps_scale / sum_eps
      budget_delta = 0
      if sum_delta != 0:
        budget_delta = self.delta * budget._delta_scale / sum_delta
      budget.set_eps_delta(budget_eps, budget_delta)
    self.print_results()

  def print_results(self):
    print("Computed budget")
    for budget in self.requested_budgets:
      print(f"{budget._text} : eps={budget.eps} delta={budget.delta}")
