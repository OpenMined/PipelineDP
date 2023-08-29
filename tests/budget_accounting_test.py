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
"""Budget Accounting Test"""

import sys
import unittest
from dataclasses import dataclass
# TODO: import only modules https://google.github.io/styleguide/pyguide.html#22-imports
from pipeline_dp.budget_accounting import MechanismSpec
from pipeline_dp.aggregate_params import MechanismType
from pipeline_dp.budget_accounting import NaiveBudgetAccountant
from pipeline_dp.budget_accounting import PLDBudgetAccountant
from absl.testing import parameterized


# pylint: disable=protected-access
class NaiveBudgetAccountantTest(parameterized.TestCase):

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

    def test_two_calls_compute_budgets_raise_exception(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-6)
        budget_accountant.request_budget(mechanism_type=MechanismType.LAPLACE)
        budget_accountant.compute_budgets()
        with self.assertRaises(Exception):
            # Budget can be computed only once.
            budget_accountant.compute_budgets()

    def test_request_after_compute_raise_exception(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-6)
        budget_accountant.request_budget(mechanism_type=MechanismType.LAPLACE)
        budget_accountant.compute_budgets()
        with self.assertRaises(Exception):
            # Budget can not be requested after it has been already computed.
            budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE)

    @parameterized.parameters(1, 2, 10)
    def test_num_aggregations(self, num_aggregations):
        total_epsilon, total_delta = 1, 1e-6
        budget_accountant = NaiveBudgetAccountant(
            total_epsilon=total_epsilon,
            total_delta=total_delta,
            num_aggregations=num_aggregations)
        for _ in range(num_aggregations):
            budget = budget_accountant._compute_budget_for_aggregation(1)
            expected_epsilon = total_epsilon / num_aggregations
            expected_delta = total_delta / num_aggregations
            self.assertAlmostEqual(expected_epsilon, budget.epsilon)
            self.assertAlmostEqual(expected_delta, budget.delta)

        budget_accountant.compute_budgets()

    def test_aggregation_weights(self):

        total_epsilon, total_delta = 1, 1e-6
        weights = [1, 2, 5]
        budget_accountant = NaiveBudgetAccountant(total_epsilon=total_epsilon,
                                                  total_delta=total_delta,
                                                  aggregation_weights=weights)
        for weight in weights:
            budget = budget_accountant._compute_budget_for_aggregation(weight)
            expected_epsilon = total_epsilon * weight / sum(weights)
            expected_delta = total_delta * weight / sum(weights)
            self.assertAlmostEqual(expected_epsilon, budget.epsilon)
            self.assertAlmostEqual(expected_delta, budget.delta)

        budget_accountant.compute_budgets()

    @parameterized.parameters(True, False)
    def test_not_enough_aggregations(self, use_num_aggregations):
        weights = num_aggregations = None
        if use_num_aggregations:
            num_aggregations = 2
        else:
            weights = [1, 1]  # 2 aggregations
        budget_accountant = NaiveBudgetAccountant(
            total_epsilon=1,
            total_delta=1e-6,
            num_aggregations=num_aggregations,
            aggregation_weights=weights)

        budget_accountant._compute_budget_for_aggregation(1)
        with self.assertRaises(ValueError):
            # num_aggregations = 2, but only 1 aggregation_scope was created
            budget_accountant.compute_budgets()


@unittest.skipIf(sys.version_info.major == 3 and sys.version_info.minor <= 8,
                 "dp_accounting library only support python >=3.9")
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

    def test_two_calls_compute_budgets_raise_exception(self):
        budget_accountant = PLDBudgetAccountant(total_epsilon=1,
                                                total_delta=1e-6)
        budget_accountant.request_budget(mechanism_type=MechanismType.LAPLACE)
        budget_accountant.compute_budgets()
        with self.assertRaises(Exception):
            # Budget can be computed only once.
            budget_accountant.compute_budgets()

    def test_request_after_compute_raise_exception(self):
        budget_accountant = PLDBudgetAccountant(total_epsilon=1,
                                                total_delta=1e-6)
        budget_accountant.request_budget(mechanism_type=MechanismType.LAPLACE)
        budget_accountant.compute_budgets()
        with self.assertRaises(Exception):
            # Budget can not be requested after it has been already computed.
            budget_accountant.request_budget(
                mechanism_type=MechanismType.LAPLACE)

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
                                  epsilon=0.168,
                                  delta=1e-3,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=10,
                                          expected_noise_std=49.872,
                                          mechanism_type=MechanismType.LAPLACE,
                                          weight=1,
                                          sensitivity=1)
                                  ],
                                  expected_pipeline_noise_std=49.872),
            ComputeBudgetTestCase(name="gaussian_mechanisms",
                                  epsilon=0.115,
                                  delta=1e-3,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=10,
                                          expected_noise_std=50.25,
                                          mechanism_type=MechanismType.GAUSSIAN,
                                          weight=1,
                                          sensitivity=1)
                                  ],
                                  expected_pipeline_noise_std=50.25),
            ComputeBudgetTestCase(
                name="multiple_noise_kinds",
                epsilon=0.240,
                delta=1e-3,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=49.73,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=1,
                        sensitivity=1),
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=49.73,
                        mechanism_type=MechanismType.GAUSSIAN,
                        weight=1,
                        sensitivity=1),
                    ComputeBudgetMechanisms(
                        count=5,
                        expected_noise_std=49.73,
                        mechanism_type=MechanismType.GENERIC,
                        weight=1,
                        sensitivity=1,
                        expected_mechanism_epsilon=0.02838875378244,
                        expected_mechanism_delta=0.00010439193305478515)
                ],
                expected_pipeline_noise_std=49.73),
            ComputeBudgetTestCase(name="multiple_weights",
                                  epsilon=1.873,
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
                                  epsilon=0.246,
                                  delta=1e-5,
                                  mechanisms=[
                                      ComputeBudgetMechanisms(
                                          count=6,
                                          expected_noise_std=40.048,
                                          mechanism_type=MechanismType.LAPLACE,
                                          weight=1,
                                          sensitivity=2),
                                      ComputeBudgetMechanisms(
                                          count=2,
                                          expected_noise_std=80.096,
                                          mechanism_type=MechanismType.GAUSSIAN,
                                          weight=1,
                                          sensitivity=4)
                                  ],
                                  expected_pipeline_noise_std=20.024),
            ComputeBudgetTestCase(name="multiple_weights_and_sensitivities",
                                  epsilon=0.719,
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
                epsilon=0.822,
                delta=1e-6,
                mechanisms=[
                    ComputeBudgetMechanisms(
                        count=4,
                        expected_noise_std=20.01,
                        mechanism_type=MechanismType.LAPLACE,
                        weight=4,
                        sensitivity=2),
                    ComputeBudgetMechanisms(
                        count=6,
                        expected_noise_std=80.04,
                        mechanism_type=MechanismType.GAUSSIAN,
                        weight=2,
                        sensitivity=4),
                    ComputeBudgetMechanisms(
                        count=1,
                        expected_noise_std=80.04,
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
                expected_pipeline_noise_std=40.02)
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
                    case.epsilon,
                    actual_epsilon,
                    delta=1e-3,
                    msg=f"failed test {case.name} expected epsilon "
                    f"{case.epsilon} got {actual_epsilon}")
            accountant.compute_budgets()
            self.assertAlmostEqual(
                first=case.expected_pipeline_noise_std,
                second=accountant.minimum_noise_std,
                delta=1e-2,
                msg=f"failed test {case.name} expected pipeline noise "
                f"{case.expected_pipeline_noise_std} "
                f"got {accountant.minimum_noise_std}")
            for mechanism_expectations in actual_mechanisms:
                expected_mechanism_noise_std, expected_mechanism_epsilon, expected_mechanism_delta, actual_mechanism = mechanism_expectations
                self.assertAlmostEqual(
                    first=expected_mechanism_noise_std,
                    second=actual_mechanism.noise_standard_deviation,
                    delta=1e-2,
                    msg=f"failed test {case.name} expected mechanism noise "
                    f"{expected_mechanism_noise_std} "
                    f"got {actual_mechanism.noise_standard_deviation}")
                if actual_mechanism.mechanism_type == MechanismType.GENERIC:
                    self.assertAlmostEqual(
                        first=expected_mechanism_epsilon,
                        second=actual_mechanism._eps,
                        delta=1e-3,
                        msg=f"failed test {case.name} expected mechanism epsilon "
                        f"{expected_mechanism_epsilon} "
                        f"got {actual_mechanism._eps}")
                    self.assertAlmostEqual(
                        first=expected_mechanism_delta,
                        second=actual_mechanism._delta,
                        delta=1e-3,
                        msg=f"failed test {case.name} expected mechanism delta "
                        f"{expected_mechanism_delta} "
                        f"got {actual_mechanism._delta}")


if __name__ == '__main__':
    unittest.main()
