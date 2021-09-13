"""Budget Accounting Test"""

import unittest
from dataclasses import dataclass
from pipeline_dp.aggregate_params import NoiseKind
from pipeline_dp.budget_accounting import MechanismSpec, NaiveBudgetAccountant, PLDBudgetAccountant


# pylint: disable=protected-access
class NaiveBudgetAccountantTest(unittest.TestCase):

    def test_validation(self):
        NaiveBudgetAccountant(total_epsilon=1, total_delta=1e-10)  # No exception.
        NaiveBudgetAccountant(total_epsilon=1, total_delta=0)  # No exception.

        with self.assertRaises(ValueError):
            NaiveBudgetAccountant(
                total_epsilon=0, total_delta=1e-10)  # Epsilon must be positive.

        with self.assertRaises(ValueError):
            NaiveBudgetAccountant(
                total_epsilon=0.5, total_delta=-1e-10)  # Delta must be non-negative.

    def test_request_budget(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1, total_delta=0)
        budget = budget_accountant.request_budget(noise_kind=NoiseKind.LAPLACE)
        self.assertTrue(budget)  # An object must be returned.

        with self.assertRaises(AssertionError):
            print(budget.eps)  # The privacy budget is not calculated yet.

        with self.assertRaises(AssertionError):
            print(budget.delta)  # The privacy budget is not calculated yet.

    def test_compute_budgets(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1, total_delta=1e-6)
        budget1 = budget_accountant.request_budget(noise_kind=NoiseKind.LAPLACE)
        budget2 = budget_accountant.request_budget(noise_kind=NoiseKind.GAUSSIAN,
                                                   weight=3)
        budget_accountant.compute_budgets()

        self.assertEqual(budget1.eps, 0.25)
        self.assertEqual(budget1.delta,
                         0)  # Delta should be 0 if mechanism is Gaussian.

        self.assertEqual(budget2.eps, 0.75)
        self.assertEqual(budget2.delta, 1e-6)


class PLDBudgetAccountantTest(unittest.TestCase):

    def test_noise_not_calculated(self):
        with self.assertRaises(AssertionError):
            mechanism = MechanismSpec(NoiseKind.LAPLACE)
            print(mechanism.noise_standard_deviation())

    def test_invalid_epsilon(self):
        with self.assertRaises(ValueError):
            PLDBudgetAccountant(total_epsilon=0, total_delta=1e-5)

    def test_invalid_delta(self):
        with self.assertRaises(ValueError):
            PLDBudgetAccountant(total_epsilon=1, total_delta=-1e-5)

    def test_invalid_gaussian_delta(self):
        accountant = PLDBudgetAccountant(total_epsilon=1, total_delta=0)
        with self.assertRaises(AssertionError):
            accountant.request_budget(NoiseKind.GAUSSIAN)

    def test_compute_budgets_none_noise(self):
        accountant = PLDBudgetAccountant(total_epsilon=3, total_delta=1e-5)
        accountant.compute_budgets()
        self.assertEqual(None, accountant.minimum_noise_std)

    def test_compute_budgets(self):

        @dataclass
        class ComputeBudgetMechanisms:
            count: int
            expected_noise_std: float
            noise_kind: NoiseKind
            weight: float
            sensitivity: float

        @dataclass
        class ComputeBudgetTestCase:
            name: str
            epsilon: float
            delta: float
            expected_pipeline_noise_std: float
            mechanisms: []

        testcases = [
            ComputeBudgetTestCase(
                name="standard_laplace",
                epsilon=4,
                delta=0,
                mechanisms=[
                    ComputeBudgetMechanisms(2, 0.7071067811865476,
                                            NoiseKind.LAPLACE, 1, 1)
                ],
                expected_pipeline_noise_std=0.7071067811865476),
            ComputeBudgetTestCase(
                name="standard_laplace_weights",
                epsilon=4,
                delta=0,
                mechanisms=[
                    ComputeBudgetMechanisms(2, 0.7071067811865476,
                                            NoiseKind.LAPLACE, 2, 1)
                ],
                expected_pipeline_noise_std=1.4142135623730951),
            ComputeBudgetTestCase(
                name="standard_laplace_sensitivities",
                epsilon=3,
                delta=0,
                mechanisms=[
                    ComputeBudgetMechanisms(2, 2.82842712474619,
                                            NoiseKind.LAPLACE, 1, 3)
                ],
                expected_pipeline_noise_std=0.9428090415820634),
            ComputeBudgetTestCase(name="laplace_mechanisms",
                                  epsilon=0.19348133991361996,
                                  delta=1e-3,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          10, 50, NoiseKind.LAPLACE, 1, 1)
                                  ],
                                  expected_pipeline_noise_std=50),
            ComputeBudgetTestCase(name="gaussian_mechanisms",
                                  epsilon=0.16421041618759222,
                                  delta=1e-3,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          10, 50, NoiseKind.GAUSSIAN, 1, 1)
                                  ],
                                  expected_pipeline_noise_std=50),
            ComputeBudgetTestCase(
                name="multiple_noise_kinds",
                epsilon=0.17915869168056622,
                delta=1e-3,
                mechanisms=[
                    ComputeBudgetMechanisms(5, 50, NoiseKind.LAPLACE, 1, 1),
                    ComputeBudgetMechanisms(5, 50, NoiseKind.GAUSSIAN, 1, 1)
                ],
                expected_pipeline_noise_std=50),
            ComputeBudgetTestCase(
                name="multiple_weights",
                epsilon=1.924852037917208,
                delta=1e-5,
                mechanisms=[
                    ComputeBudgetMechanisms(4, 10, NoiseKind.LAPLACE, 2, 1),
                    ComputeBudgetMechanisms(4, 5, NoiseKind.GAUSSIAN, 4, 1)
                ],
                expected_pipeline_noise_std=20),
            ComputeBudgetTestCase(
                name="multiple_sensitivities",
                epsilon=0.2764312848667339,
                delta=1e-5,
                mechanisms=[
                    ComputeBudgetMechanisms(6, 40, NoiseKind.LAPLACE, 1, 2),
                    ComputeBudgetMechanisms(2, 80, NoiseKind.GAUSSIAN, 1, 4)
                ],
                expected_pipeline_noise_std=20),
            ComputeBudgetTestCase(
                name="multiple_weights_and_sensitivities",
                epsilon=0.780797891312483,
                delta=1e-5,
                mechanisms=[
                    ComputeBudgetMechanisms(4, 10, NoiseKind.LAPLACE, 4, 2),
                    ComputeBudgetMechanisms(6, 40, NoiseKind.GAUSSIAN, 2, 4)
                ],
                expected_pipeline_noise_std=20),
            ComputeBudgetTestCase(
                name="multiple_weights_and_sensitivities_variants",
                epsilon=0.9165937807680077,
                delta=1e-6,
                mechanisms=[
                    ComputeBudgetMechanisms(4, 20, NoiseKind.LAPLACE, 4, 2),
                    ComputeBudgetMechanisms(6, 80, NoiseKind.GAUSSIAN, 2, 4),
                    ComputeBudgetMechanisms(1, 80, NoiseKind.GAUSSIAN, 3, 6),
                    ComputeBudgetMechanisms(5, 15, NoiseKind.LAPLACE, 8, 3),
                ],
                expected_pipeline_noise_std=40)
        ]

        for case in testcases:
            accountant = PLDBudgetAccountant(case.epsilon, case.delta, 1e-2)
            actual_mechanisms = []
            for mechanism in case.mechanisms:
                for _ in range(0, mechanism.count):
                    actual_mechanisms.append(
                        (mechanism.expected_noise_std,
                         accountant.request_budget(
                             mechanism.noise_kind,
                             weight=mechanism.weight,
                             sensitivity=mechanism.sensitivity)))
            self.assertEqual(
                len(actual_mechanisms), len(accountant._mechanisms),
                f"failed test {case.name} expected len {len(actual_mechanisms)} "
                f"got len {len(accountant._mechanisms)}")
            if case.delta > 0:
                compare_pld = accountant._compose_distributions(
                    case.expected_pipeline_noise_std)
                actual_epsilon = compare_pld.get_epsilon_for_delta(case.delta)
                self.assertAlmostEqual(
                    case.epsilon, actual_epsilon, 3,
                    f"failed test {case.name} expected epsilon {case.epsilon} got {actual_epsilon}"
                )
            accountant.compute_budgets()
            self.assertEqual(
                case.expected_pipeline_noise_std, accountant.minimum_noise_std,
                f"failed test {case.name} expected pipeline noise {case.expected_pipeline_noise_std} "
                f"got {accountant.minimum_noise_std}")
            for actual_mechanism_tuple in actual_mechanisms:
                expected_mechanism_noise_std, actual_mechanism = actual_mechanism_tuple
                self.assertEqual(
                    expected_mechanism_noise_std,
                    actual_mechanism.noise_standard_deviation,
                    f"failed test {case.name} expected mechanism noise {expected_mechanism_noise_std} "
                    f"got {actual_mechanism.noise_standard_deviation}")


if __name__ == '__main__':
    unittest.main()
