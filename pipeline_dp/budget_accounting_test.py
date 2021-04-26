import unittest

import pipeline_dp


class BudgetAccountant(unittest.TestCase):

  def test_validation(self):
    pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10)  # no exception
    pipeline_dp.BudgetAccountant(epsilon=1, delta=0)  # no exception

    with self.assertRaises(ValueError):
      # epsilon must be positive
      pipeline_dp.BudgetAccountant(epsilon=0, delta=1e-10)

    with self.assertRaises(ValueError):
      # epsilon must be positive
      pipeline_dp.BudgetAccountant(
          epsilon=0.5, delta=-1e-10)  # Delta must be non-negative


if __name__ == '__main__':
  unittest.main()
