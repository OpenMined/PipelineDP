import unittest

import pipeline_dp


class BudgetAccountantTest(unittest.TestCase):

  def test_validation(self):
    pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10)  # No exception.
    pipeline_dp.BudgetAccountant(epsilon=1, delta=0)  # No exception.

    with self.assertRaises(ValueError):
      pipeline_dp.BudgetAccountant(epsilon=0, delta=1e-10)  # Epsilon must be positive.

    with self.assertRaises(ValueError):
      pipeline_dp.BudgetAccountant(epsilon=0.5, delta=-1e-10)  # Delta must be non-negative.

  def test_request_budget(self):
    budget_accountant = pipeline_dp.BudgetAccountant(epsilon=1, delta=0)
    budget = budget_accountant.request_budget(1, use_eps=False, use_delta=False)
    self.assertTrue(budget)  # An object must be returned.

    with self.assertRaises(AssertionError):
      print(budget.eps)  # The privacy budget is not calculated yet.

    with self.assertRaises(AssertionError):
      print(budget.delta)  # The privacy budget is not calculated yet.

  def test_compute_budgets(self):
    budget_accountant = pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-6)
    budget1 = budget_accountant.request_budget(1, use_eps=True, use_delta=False)
    budget2 = budget_accountant.request_budget(3, use_eps=True, use_delta=True)
    budget_accountant.compute_budgets()

    self.assertEqual(budget1.eps, 0.25)
    self.assertEqual(budget1.delta, 0)  # Delta should be 0 if use_delta is False.

    self.assertEqual(budget2.eps, 0.75)
    self.assertEqual(budget2.delta, 1e-6)


if __name__ == '__main__':
  unittest.main()
