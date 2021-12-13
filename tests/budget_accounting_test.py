"""Budget Accounting Test"""

import unittest
from dataclasses import dataclass
# TODO: import only modules https://google.github.io/styleguide/pyguide.html#22-imports
from pipeline_dp.budget_accounting import MechanismSpec
from pipeline_dp.aggregate_params import MechanismType
from pipeline_dp.budget_accounting import NaiveBudgetAccountant
from pipeline_dp.budget_accounting import PLDBudgetAccountant


# pylint: disable=protected-access
class NaiveBudgetAccountantTest(unittest.TestCase):

    def test_validation(self):
        NaiveBudgetAccountant(total_epsilon=1,
                              total_delta=1e-10)  # No exception.
        NaiveBudgetAccountant(total_epsilon=1, total_delta=0)  # No exception.

        with self.assertRaises(ValueError):
            NaiveBudgetAccountant(
                total_epsilon=0, total_delta=1e-10)  # Epsilon must be positive.

        with self.assertRaises(ValueError):
            NaiveBudgetAccountant(
                total_epsilon=0.5,
                total_delta=-1e-10)  # Delta must be non-negative.

    def test_request_budget(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=0)
        budget = budget_accountant.request_budget(
            mechanism_type=MechanismType.LAPLACE)
        self.assertTrue(budget)  # An object must be returned.

        with self.assertRaises(AssertionError):
            print(budget.eps)  # The privacy budget is not calculated yet.

        with self.assertRaises(AssertionError):
            print(budget.delta)  # The privacy budget is not calculated yet.

    def test_compute_budgets(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-6)
        budget1 = budget_accountant.request_budget(
            mechanism_type=MechanismType.LAPLACE)
        budget2 = budget_accountant.request_budget(
            mechanism_type=MechanismType.GAUSSIAN, weight=3)
        budget_accountant.compute_budgets()

        self.assertEqual(budget1.eps, 0.25)
        self.assertEqual(budget1.delta,
                         0)  # Delta should be 0 if mechanism is Laplace.

        self.assertEqual(budget2.eps, 0.75)
        self.assertEqual(budget2.delta, 1e-6)

    def test_budget_scopes(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-6)

        with budget_accountant.scope(weight=0.4):
            budget1 = budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE)
            budget2 = budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE, weight=3)

        with budget_accountant.scope(weight=0.6):
            budget3 = budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE)
            budget4 = budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE, weight=4)

        budget_accountant.compute_budgets()

        self.assertEqual(budget1.eps, 0.4 * (1 / 4))
        self.assertEqual(budget2.eps, 0.4 * (3 / 4))
        self.assertEqual(budget3.eps, 0.6 * (1 / 5))
        self.assertEqual(budget4.eps, 0.6 * (4 / 5))

    def test_budget_scopes_no_parentscope(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-6)

        # Allocated in the top-level scope with no weight specified
        budget1 = budget_accountant.request_budget(
            mechanism_type=MechanismType.LAPLACE)

        with budget_accountant.scope(weight=0.5):
            budget2 = budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE)

        budget_accountant.compute_budgets()

        self.assertEqual(budget1.eps, 1.0 / (1.0 + 0.5))
        self.assertEqual(budget2.eps, 0.5 / (1.0 + 0.5))

    def test_count(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-6)
        budget1 = budget_accountant.request_budget(
            mechanism_type=MechanismType.LAPLACE, weight=4)
        budget2 = budget_accountant.request_budget(
            mechanism_type=MechanismType.GAUSSIAN, weight=3, count=2)
        budget_accountant.compute_budgets()

        self.assertEqual(budget1.eps, 0.4)
        self.assertEqual(budget1.delta,
                         0)  # Delta should be 0 if mechanism is Laplace.

        self.assertEqual(budget2.eps, 0.3)
        self.assertEqual(budget2.delta, 5e-7)


class PLDBudgetAccountantTest(unittest.TestCase):

    def test_noise_not_calculated(self):
        with self.assertRaises(AssertionError):
            mechanism = MechanismSpec(MechanismType.LAPLACE)
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
            accountant.request_budget(MechanismType.GAUSSIAN)

    def test_compute_budgets_none_noise(self):
        accountant = PLDBudgetAccountant(total_epsilon=3, total_delta=1e-5)
        accountant.compute_budgets()
        self.assertEqual(None, accountant.minimum_noise_std)

    def test_compute_budgets(self):

        @dataclass
        class ComputeBudgetMechanisms:
            count: int
            expected_noise_std: float
            mechanism_type: MechanismType
            weight: float
            sensitivity: float
            expected_mechanism_epsilon: float = None
            expected_mechanism_delta: float = None

        @dataclass
        class ComputeBudgetTestCase:
            name: str
            epsilon: float
            delta: float
            expected_pipeline_noise_std: float
            mechanisms: []

        testcases = [
            ComputeBudgetTestCase(
                name="generic",
                epsilon=0.22999925338484556,
                delta=1e-5,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=1,
                        expected_noise_std=6.41455078125,
                        mechanism_type=MechanismType.GENERIC,
                        weight=1,
                        sensitivity=1,
                        expected_mechanism_epsilon=0.2204717161227536,
                        expected_mechanism_delta=9.585757904781109e-06)
                ],
                expected_pipeline_noise_std=6.41455078125),
            ComputeBudgetTestCase(
                name="generic_multiple",
                epsilon=0.6599974547358093,
                delta=1e-5,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=3,
                        expected_noise_std=6.71649169921875,
                        mechanism_type=MechanismType.GENERIC,
                        weight=1,
                        sensitivity=1,
                        expected_mechanism_epsilon=0.21055837268995567,
                        expected_mechanism_delta=3.190290677321479e-06)
                ],
                expected_pipeline_noise_std=6.71649169921875),
            ComputeBudgetTestCase(
                name="standard_laplace",
                epsilon=4,
                delta=0,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=2,
                        expected_noise_std=0.7071067811865476,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=1,
                        sensitivity=1)
                ],
                expected_pipeline_noise_std=0.7071067811865476),
            ComputeBudgetTestCase(
                name="standard_laplace_weights",
                epsilon=4,
                delta=0,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=2,
                        expected_noise_std=0.7071067811865476,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=2,
                        sensitivity=1)
                ],
                expected_pipeline_noise_std=1.4142135623730951),
            ComputeBudgetTestCase(
                name="standard_laplace_sensitivities",
                epsilon=3,
                delta=0,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=2,
                        expected_noise_std=2.82842712474619,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=1,
                        sensitivity=3)
                ],
                expected_pipeline_noise_std=0.9428090415820634),
            ComputeBudgetTestCase(name="laplace_mechanisms",
                                  epsilon=0.19348133991361996,
                                  delta=1e-3,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=10,
                                          expected_noise_std=50,
                                          mechanism_type=MechanismType.LAPLACE,
                                          weight=1,
                                          sensitivity=1)
                                  ],
                                  expected_pipeline_noise_std=50),
            ComputeBudgetTestCase(name="gaussian_mechanisms",
                                  epsilon=0.16421041618759222,
                                  delta=1e-3,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=10,
                                          expected_noise_std=50,
                                          mechanism_type=MechanismType.GAUSSIAN,
                                          weight=1,
                                          sensitivity=1)
                                  ],
                                  expected_pipeline_noise_std=50),
            ComputeBudgetTestCase(
                name="multiple_noise_kinds",
                epsilon=0.2719439419475211,
                delta=1e-3,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=49.81597900390625,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=1,
                        sensitivity=1),
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=49.81597900390625,
                        mechanism_type=MechanismType.GAUSSIAN,
                        weight=1,
                        sensitivity=1),
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=49.81597900390625,
                        mechanism_type=MechanismType.GENERIC,
                        weight=1,
                        sensitivity=1,
                        expected_mechanism_epsilon=0.02838875378244,
                        expected_mechanism_delta=0.00010439193305478515)
                ],
                expected_pipeline_noise_std=49.81597900390625),
            ComputeBudgetTestCase(name="multiple_weights",
                                  epsilon=1.924852037917208,
                                  delta=1e-5,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=4,
                                          expected_noise_std=10,
                                          mechanism_type=MechanismType.LAPLACE,
                                          weight=2,
                                          sensitivity=1),
                                      ComputeBudgetMechanisms(
                                          count=4,
                                          expected_noise_std=5,
                                          mechanism_type=MechanismType.GAUSSIAN,
                                          weight=4,
                                          sensitivity=1)
                                  ],
                                  expected_pipeline_noise_std=20),
            ComputeBudgetTestCase(name="multiple_sensitivities",
                                  epsilon=0.2764312848667339,
                                  delta=1e-5,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=6,
                                          expected_noise_std=40,
                                          mechanism_type=MechanismType.LAPLACE,
                                          weight=1,
                                          sensitivity=2),
                                      ComputeBudgetMechanisms(
                                          count=2,
                                          expected_noise_std=80,
                                          mechanism_type=MechanismType.GAUSSIAN,
                                          weight=1,
                                          sensitivity=4)
                                  ],
                                  expected_pipeline_noise_std=20),
            ComputeBudgetTestCase(name="multiple_weights_and_sensitivities",
                                  epsilon=0.780797891312483,
                                  delta=1e-5,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=4,
                                          expected_noise_std=10,
                                          mechanism_type=MechanismType.LAPLACE,
                                          weight=4,
                                          sensitivity=2),
                                      ComputeBudgetMechanisms(
                                          count=6,
                                          expected_noise_std=40,
                                          mechanism_type=MechanismType.GAUSSIAN,
                                          weight=2,
                                          sensitivity=4)
                                  ],
                                  expected_pipeline_noise_std=20),
            ComputeBudgetTestCase(
                name="multiple_weights_and_sensitivities_variants",
                epsilon=0.9165937807680077,
                delta=1e-6,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=4,
                        expected_noise_std=20,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=4,
                        sensitivity=2),
                    ComputeBudgetMechanisms(
                        count=6,
                        expected_noise_std=80,
                        mechanism_type=MechanismType.GAUSSIAN,
                        weight=2,
                        sensitivity=4),
                    ComputeBudgetMechanisms(
                        count=1,
                        expected_noise_std=80,
                        mechanism_type=MechanismType.GAUSSIAN,
                        weight=3,
                        sensitivity=6),
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=15,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=8,
                        sensitivity=3),
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
                         mechanism.expected_mechanism_epsilon,
                         mechanism.expected_mechanism_delta,
                         accountant.request_budget(
                             mechanism.mechanism_type,
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
            self.assertAlmostEqual(
                first=case.expected_pipeline_noise_std,
                second=accountant.minimum_noise_std,
                places=3,
                msg=
                f"failed test {case.name} expected pipeline noise {case.expected_pipeline_noise_std} "
                f"got {accountant.minimum_noise_std}")
            for mechanism_expectations in actual_mechanisms:
                expected_mechanism_noise_std, expected_mechanism_epsilon, expected_mechanism_delta, actual_mechanism = mechanism_expectations
                self.assertAlmostEqual(
                    first=expected_mechanism_noise_std,
                    second=actual_mechanism.noise_standard_deviation,
                    places=3,
                    msg=
                    f"failed test {case.name} expected mechanism noise {expected_mechanism_noise_std} "
                    f"got {actual_mechanism.noise_standard_deviation}")
                if actual_mechanism.mechanism_type == MechanismType.GENERIC:
                    self.assertAlmostEqual(
                        first=expected_mechanism_epsilon,
                        second=actual_mechanism._eps,
                        places=3,
                        msg=
                        f"failed test {case.name} expected mechanism epsilon {expected_mechanism_epsilon} "
                        f"got {actual_mechanism._eps}")
                    self.assertAlmostEqual(
                        first=expected_mechanism_delta,
                        second=actual_mechanism._delta,
                        places=3,
                        msg=
                        f"failed test {case.name} expected mechanism delta {expected_mechanism_delta} "
                        f"got {actual_mechanism._delta}")


if __name__ == '__main__':
    unittest.main()
