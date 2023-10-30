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
import unittest
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from typing import Iterable, Optional
from scipy import stats
import math
from unittest.mock import patch
from unittest.mock import MagicMock

import pipeline_dp
import pipeline_dp.dp_computations as dp_computations
from pipeline_dp.aggregate_params import NoiseKind
from pipeline_dp import aggregate_params
from pipeline_dp import budget_accounting

N_ITERATIONS = 200000
DUMMY_MIN_VALUE = 2.0
DUMMY_MAX_VALUE = 3.0


class DPComputationsTest(parameterized.TestCase):

    def test_l0_sensitivity(self):
        params = dp_computations.ScalarNoiseParams(
            eps=1,
            delta=1e-10,
            min_value=2,
            max_value=3,
            min_sum_per_partition=None,
            max_sum_per_partition=None,
            max_partitions_contributed=4,
            max_contributions_per_partition=5,
            noise_kind=NoiseKind.LAPLACE)
        self.assertEqual(params.l0_sensitivity(), 4)

    def test_l1_sensitivity(self):
        self.assertAlmostEqual(dp_computations.compute_l1_sensitivity(
            l0_sensitivity=4.5, linf_sensitivity=12.123),
                               54.5535,
                               delta=0.1)

    def test_l2_sensitivity(self):
        self.assertAlmostEqual(dp_computations.compute_l2_sensitivity(
            l0_sensitivity=4.5, linf_sensitivity=12.123),
                               25.716766525,
                               delta=0.1)

    def test_compute_sigma(self):
        self.assertEqual(
            114.375,
            dp_computations.compute_sigma(eps=0.5,
                                          delta=1e-10,
                                          l2_sensitivity=10))

    def _test_laplace_kolmogorov_smirnov(self, num_trials: int, results,
                                         expected_mean: float,
                                         expected_beta: float):
        laplace_sample = np.random.laplace(expected_mean, expected_beta,
                                           num_trials)
        statistic, pvalue = stats.ks_2samp(results, laplace_sample)
        # By the p-value definition, the next line has flakiness 1e-4.
        self.assertGreaterEqual(pvalue, 1e-4)

    def _test_gaussian_kolmogorov_smirnov(self, num_trials: int, results,
                                          expected_mean: float,
                                          expected_sigma: float):
        guassian_sample = np.random.normal(expected_mean, expected_sigma,
                                           num_trials)
        statistic, pvalue = stats.ks_2samp(results, guassian_sample)
        # By the p-value definition, the next line has flakiness 1e-4.
        self.assertGreaterEqual(pvalue, 1e-4)

    def _laplace_prob_mass_within_one_std(self):
        return 1.0 - math.exp(-math.sqrt(2.0))

    def _laplace_prob_mass_one_to_two_stds(self):
        return math.exp(-math.sqrt(2.0)) - math.exp(-2.0 * math.sqrt(2.0))

    def _gaussian_prob_mass_within_one_std(self):
        return 0.68268949213

    def _gaussian_prob_mass_one_to_two_stds(self):
        return 0.27181024396

    def _test_samples_from_distribution(self, num_trials: int, values,
                                        expected_mean: float,
                                        expected_sigma: float,
                                        prob_mass_within_one_std: float,
                                        prob_mass_one_to_two_stds: float):
        num_results_within_one_std = 0
        plus_one_std = expected_mean + expected_sigma
        minus_one_std = expected_mean - expected_sigma
        num_results_one_to_two_stds = 0
        plus_two_std = expected_mean + 2.0 * expected_sigma
        minus_two_std = expected_mean - 2 * expected_sigma

        for x in values:
            if minus_one_std <= x <= plus_one_std:
                num_results_within_one_std += 1
            elif minus_two_std <= x <= plus_two_std:
                num_results_one_to_two_stds += 1

        # 99.993% confidence interval (probability normal 4 sigma from mean).
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        num_trials_double = 1.0 * num_trials
        self.assertAlmostEqual(
            num_results_within_one_std / num_trials_double,
            prob_mass_within_one_std,
            delta=4.0 * math.sqrt(prob_mass_within_one_std *
                                  (1 - prob_mass_within_one_std) / num_trials))
        self.assertAlmostEqual(
            num_results_one_to_two_stds / num_trials_double,
            prob_mass_one_to_two_stds,
            delta=4.0 * math.sqrt(prob_mass_one_to_two_stds *
                                  (1 - prob_mass_one_to_two_stds) / num_trials))
        return 0

    def _test_laplace_noise(self, num_trials: int, results,
                            expected_mean: float, l1_sensitivity: float,
                            eps: float):
        expected_beta = l1_sensitivity / eps
        self._test_samples_from_distribution(
            values=results,
            num_trials=num_trials,
            expected_mean=expected_mean,
            expected_sigma=math.sqrt(2) * expected_beta,
            prob_mass_within_one_std=self._laplace_prob_mass_within_one_std(),
            prob_mass_one_to_two_stds=self._laplace_prob_mass_one_to_two_stds())
        self._test_laplace_kolmogorov_smirnov(num_trials, results,
                                              expected_mean, expected_beta)

    def _test_gaussian_noise(self, num_trials: int, results,
                             expected_mean: float, l2_sensitivity: float,
                             eps: float, delta: float):
        expected_sigma = dp_computations.compute_sigma(eps, delta,
                                                       l2_sensitivity)
        self._test_samples_from_distribution(
            values=results,
            num_trials=num_trials,
            expected_mean=expected_mean,
            expected_sigma=expected_sigma,
            prob_mass_within_one_std=self._gaussian_prob_mass_within_one_std(),
            prob_mass_one_to_two_stds=self._gaussian_prob_mass_one_to_two_stds(
            ))
        self._test_gaussian_kolmogorov_smirnov(num_trials, results,
                                               expected_mean, expected_sigma)

    def _not_all_integers(self, results: Iterable[float]):
        return any(map(lambda x: not x.is_integer(), results))

    def test_apply_laplace_mechanism(self):
        results = [
            dp_computations.apply_laplace_mechanism(value=20,
                                                    eps=0.5,
                                                    l1_sensitivity=1)
            for _ in range(1000000)
        ]
        self.assertTrue(self._not_all_integers(results))
        self._test_laplace_noise(results=results,
                                 num_trials=1000000,
                                 expected_mean=20,
                                 eps=0.5,
                                 l1_sensitivity=1)

    @patch('pydp.algorithms.numerical_mechanisms.LaplaceMechanism')
    def test_secure_laplace_noise_is_used(self, laplace_mechanism):
        # Arrange
        mock_laplace_mechanism = MagicMock()
        laplace_mechanism.return_value = mock_laplace_mechanism
        mock_laplace_mechanism.add_noise = MagicMock(
            return_value="value_with_noise")

        # Act
        anonymized_value = dp_computations.apply_laplace_mechanism(
            value=20, eps=0.5, l1_sensitivity=3)

        # Assert
        laplace_mechanism.assert_called_with(epsilon=0.5, sensitivity=3)
        mock_laplace_mechanism.add_noise.assert_called_with(20)
        self.assertEqual("value_with_noise", anonymized_value)

    def test_apply_gaussian_mechanism(self):
        results = [
            dp_computations.apply_gaussian_mechanism(value=20,
                                                     eps=0.5,
                                                     delta=1e-10,
                                                     l2_sensitivity=1)
            for _ in range(1000000)
        ]
        self.assertTrue(self._not_all_integers(results))
        self._test_gaussian_noise(results=results,
                                  num_trials=1000000,
                                  expected_mean=20,
                                  eps=0.5,
                                  delta=1e-10,
                                  l2_sensitivity=1)

    @patch('pydp.algorithms.numerical_mechanisms.GaussianMechanism')
    def test_secure_gaussian_noise_is_used(self, gaussian_mechanism):
        # Arrange
        mock_gaussian_mechanism = MagicMock()
        gaussian_mechanism.return_value = mock_gaussian_mechanism
        mock_gaussian_mechanism.add_noise = MagicMock(
            return_value="value_with_noise")

        # Act
        anonymized_value = dp_computations.apply_gaussian_mechanism(
            value=20, eps=0.5, delta=1e-10, l2_sensitivity=3)

        # Assert
        gaussian_mechanism.assert_called_with(0.5, 1e-10, 3)
        mock_gaussian_mechanism.add_noise.assert_called_with(20)
        self.assertEqual("value_with_noise", anonymized_value)

    def test_equally_split_budget(self):
        # The number of mechanisms must be bigger than 0.
        with self.assertRaises(ValueError):
            dp_computations.equally_split_budget(0.5, 1e-10, 0)

        # Only one mechanism.
        self.assertEqual(dp_computations.equally_split_budget(0.5, 1e-10, 1),
                         [(0.5, 1e-10)])

        # Multiple mechanisms.
        expected_budgets = [(0.5 / 5, 1e-10 / 5) for _ in range(4)]
        expected_budgets.append((0.5 - 4 * (0.5 / 5), 1e-10 - 4 * (1e-10 / 5)))

        self.assertEqual(dp_computations.equally_split_budget(0.5, 1e-10, 5),
                         expected_budgets)

    def test_compute_dp_variance_equal_min_max(self):
        params = dp_computations.ScalarNoiseParams(
            eps=0.5,
            delta=1e-10,
            min_value=42.0,
            max_value=42.0,  # = min_value
            min_sum_per_partition=None,
            max_sum_per_partition=None,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)

        count, sum, mean, var = dp_computations.compute_dp_var(
            count=10,
            normalized_sum=400,
            normalized_sum_squares=400,
            dp_params=params)
        self.assertEqual(mean, 42.0)
        self.assertEqual(var, 0.0)

    def test_compute_dp_var(self):
        params = dp_computations.ScalarNoiseParams(
            eps=10,
            delta=1e-10,
            min_value=1,
            max_value=20,
            min_sum_per_partition=None,
            max_sum_per_partition=None,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)

        (count_eps,
         count_delta), (_, _), (_, _) = dp_computations.equally_split_budget(
             params.eps, params.delta, 3)
        l0_sensitivity = params.l0_sensitivity()
        count_linf_sensitivity = params.max_contributions_per_partition

        expected_count = 100000
        expected_sum = 1000000
        expected_mean = 10
        expected_var = 100
        normalized_sum = -50000
        normalized_sum_squares = 10025000  # sum of squares = 20000000

        # Laplace Mechanism
        results = [
            dp_computations.compute_dp_var(
                count=expected_count,
                normalized_sum=normalized_sum,
                normalized_sum_squares=normalized_sum_squares,
                dp_params=params) for _ in range(N_ITERATIONS)
        ]
        count_values, sum_values, mean_values, var_values = zip(*results)
        self._test_laplace_noise(
            results=count_values,
            num_trials=N_ITERATIONS,
            expected_mean=100000,
            eps=count_eps,
            l1_sensitivity=dp_computations.compute_l1_sensitivity(
                l0_sensitivity, count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), expected_sum, delta=1)
        self.assertAlmostEqual(np.mean(mean_values),
                               expected_mean,
                               delta=0.00003)
        self.assertAlmostEqual(np.mean(var_values), expected_var, delta=0.1)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        results = [
            dp_computations.compute_dp_var(
                count=expected_count,
                normalized_sum=normalized_sum,
                normalized_sum_squares=normalized_sum_squares,
                dp_params=params) for _ in range(N_ITERATIONS)
        ]
        count_values, sum_values, mean_values, var_values = zip(*results)

        self._test_gaussian_noise(
            results=count_values,
            num_trials=N_ITERATIONS,
            expected_mean=100000,
            eps=count_eps,
            delta=count_delta,
            l2_sensitivity=dp_computations.compute_l2_sensitivity(
                l0_sensitivity, count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), expected_sum, delta=5)
        self.assertAlmostEqual(np.mean(mean_values),
                               expected_mean,
                               delta=0.0002)
        self.assertAlmostEqual(np.mean(var_values), expected_var, delta=0.5)

    @parameterized.parameters(
        {
            "eps": 2.0,
            "max_partitions_contributed": 10,
            "min_sum_per_partition": 0,
            "max_sum_per_partition": 3,
            "expected_std": np.sqrt(2) * (3 * 10) / 2.0
        }, {
            "eps": 3.5,
            "max_partitions_contributed": 100,
            "min_sum_per_partition": -15,
            "max_sum_per_partition": 10,
            "expected_std": np.sqrt(2) * (15 * 100) / 3.5
        })
    def test_compute_dp_sum_noise_std_laplace(self, eps: float,
                                              max_partitions_contributed: int,
                                              min_sum_per_partition: float,
                                              max_sum_per_partition: float,
                                              expected_std: float):
        params = dp_computations.ScalarNoiseParams(
            eps=eps,
            delta=0,
            min_value=None,
            max_value=None,
            max_contributions_per_partition=None,
            min_sum_per_partition=min_sum_per_partition,
            max_sum_per_partition=max_sum_per_partition,
            max_partitions_contributed=max_partitions_contributed,
            noise_kind=pipeline_dp.NoiseKind.LAPLACE)

        std = dp_computations.compute_dp_sum_noise_std(params)

        self.assertAlmostEqual(std, expected_std, delta=1e-10)

    @parameterized.parameters(
        {
            "eps": 2.0,
            "delta": 1e-8,
            "max_partitions_contributed": 2,
            "min_sum_per_partition": 5,
            "max_sum_per_partition": 10,
            "expected_std": 37.53742639189524
        }, {
            "eps": 1,
            "delta": 1e-5,
            "max_partitions_contributed": 1,
            "min_sum_per_partition": -5,
            "max_sum_per_partition": 0,
            "expected_std": 18.662109375
        })
    def test_compute_dp_sum_noise_std_gaussian(self, eps: float, delta: float,
                                               max_partitions_contributed: int,
                                               min_sum_per_partition: float,
                                               max_sum_per_partition: float,
                                               expected_std: float):
        params = dp_computations.ScalarNoiseParams(
            eps=eps,
            delta=delta,
            min_value=None,
            max_value=None,
            max_contributions_per_partition=None,
            min_sum_per_partition=min_sum_per_partition,
            max_sum_per_partition=max_sum_per_partition,
            max_partitions_contributed=max_partitions_contributed,
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)

        scale = dp_computations.compute_dp_sum_noise_std(params)

        self.assertAlmostEqual(scale, expected_std)


def create_aggregate_params(
        max_partitions_contributed: Optional[int] = None,
        max_contributions_per_partition: Optional[int] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        min_sum_per_partition: Optional[float] = None,
        max_sum_per_partition: Optional[float] = None,
        max_contributions: Optional[int] = None):
    return pipeline_dp.AggregateParams(
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=max_partitions_contributed,
        max_contributions_per_partition=max_contributions_per_partition,
        max_contributions=max_contributions,
        min_value=min_value,
        max_value=max_value,
        min_sum_per_partition=min_sum_per_partition,
        max_sum_per_partition=max_sum_per_partition)


class AdditiveMechanismTests(parameterized.TestCase):

    @parameterized.parameters(
        dict(epsilon=2, l1_sensitivity=4.5, expected_noise=2.25),
        dict(epsilon=0.1, l1_sensitivity=0.55, expected_noise=5.5),
    )
    def test_laplace_create_from_epsilon(self, epsilon, l1_sensitivity,
                                         expected_noise):
        mechanism = dp_computations.LaplaceMechanism.create_from_epsilon(
            epsilon=epsilon, l1_sensitivity=l1_sensitivity)

        self.assertEqual(mechanism.noise_kind, pipeline_dp.NoiseKind.LAPLACE)
        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise,
                               delta=1e-12)
        self.assertAlmostEqual(mechanism.std,
                               expected_noise * math.sqrt(2),
                               delta=1e-12)
        self.assertEqual(mechanism.sensitivity, l1_sensitivity)
        self.assertIsInstance(mechanism.add_noise(1000), float)

    def test_laplace_create_from_stddev(self):
        mechanism = dp_computations.LaplaceMechanism.create_from_std_deviation(
            normalized_stddev=10, l1_sensitivity=3.5)

        self.assertEqual(mechanism.noise_kind, pipeline_dp.NoiseKind.LAPLACE)
        expected_noise_parameter = 10 / np.sqrt(2) * 3.5
        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_parameter,
                               delta=1e-12)
        self.assertAlmostEqual(mechanism.std, 35)
        self.assertEqual(mechanism.sensitivity, 3.5)
        self.assertIsInstance(mechanism.add_noise(1000), float)

    @parameterized.parameters(
        dict(epsilon=2, l1_sensitivity=4.5, value=0, expected_noise_scale=2.25),
        dict(epsilon=0.1,
             l1_sensitivity=0.55,
             value=1000,
             expected_noise_scale=5.5),
        dict(epsilon=0.001,
             l1_sensitivity=10,
             value=-100,
             expected_noise_scale=10000),
    )
    def test_laplace_mechanism_distribution(self, epsilon, l1_sensitivity,
                                            value, expected_noise_scale):
        # Use Kolmogorov-Smirnov test to verify the output noise distribution.
        # https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test
        mechanism = dp_computations.LaplaceMechanism.create_from_epsilon(
            epsilon=epsilon, l1_sensitivity=l1_sensitivity)
        expected_cdf = stats.laplace(loc=value, scale=expected_noise_scale).cdf

        noised_values = [mechanism.add_noise(value) for _ in range(30000)]

        res = stats.ks_1samp(noised_values, expected_cdf)
        self.assertGreater(res.pvalue, 1e-4)

    def test_gaussian_mechanism_describe(self):
        mechanism = dp_computations.GaussianMechanism.create_from_epsilon_delta(
            epsilon=1.0, delta=1e-10, l2_sensitivity=15)
        expected = ("Gaussian mechanism:  parameter=88.06640625  eps=1.0  "
                    "delta=1e-10  l2_sensitivity=15")
        self.assertEqual(mechanism.describe(), expected)

    @parameterized.parameters(
        dict(epsilon=2,
             delta=1e-15,
             l2_sensitivity=4.5,
             expected_noise_scale=17.1826171875),
        dict(epsilon=0.1,
             delta=1e-5,
             l2_sensitivity=0.55,
             expected_noise_scale=16.9125),
    )
    def test_gaussian_mechanism_creation(self, epsilon, delta, l2_sensitivity,
                                         expected_noise_scale):
        mechanism = dp_computations.GaussianMechanism.create_from_epsilon_delta(
            epsilon=epsilon, delta=delta, l2_sensitivity=l2_sensitivity)

        self.assertEqual(mechanism.noise_kind, pipeline_dp.NoiseKind.GAUSSIAN)
        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_scale,
                               delta=1e-6)
        self.assertAlmostEqual(mechanism.std, expected_noise_scale, delta=6)
        self.assertEqual(mechanism.sensitivity, l2_sensitivity)
        self.assertIsInstance(mechanism.add_noise(1000), float)

    @parameterized.parameters(
        dict(epsilon=2,
             delta=1e-15,
             l2_sensitivity=4.5,
             value=0,
             expected_noise_scale=17.1826171875),
        dict(epsilon=0.1,
             delta=1e-5,
             l2_sensitivity=0.55,
             value=2000,
             expected_noise_scale=16.9125),
        dict(epsilon=0.2,
             delta=1e-10,
             l2_sensitivity=10,
             value=-500,
             expected_noise_scale=277.34375),
    )
    def test_gaussian_mechanism_distribution(self, epsilon, delta,
                                             l2_sensitivity, value,
                                             expected_noise_scale):
        # Use Kolmogorov-Smirnov test to verify the output noise distribution.
        # https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test
        mechanism = dp_computations.GaussianMechanism.create_from_epsilon_delta(
            epsilon=epsilon, delta=delta, l2_sensitivity=l2_sensitivity)
        self.assertEqual(mechanism.std, expected_noise_scale)

        expected_cdf = stats.norm(loc=value, scale=expected_noise_scale).cdf
        noised_values = [mechanism.add_noise(value) for _ in range(30000)]

        res = stats.ks_1samp(noised_values, expected_cdf)
        self.assertGreater(res.pvalue, 1e-4)

    def test_gaussian_create_from_stddev(self):
        mechanism = dp_computations.GaussianMechanism.create_from_std_deviation(
            normalized_stddev=5, l2_sensitivity=15)

        self.assertEqual(mechanism.noise_kind, pipeline_dp.NoiseKind.GAUSSIAN)
        self.assertEqual(mechanism.noise_parameter, 75)
        self.assertEqual(mechanism.std, 75)
        self.assertEqual(mechanism.sensitivity, 15)
        self.assertIsInstance(mechanism.add_noise(1000), float)

    def test_laplace_mechanism_describe(self):
        mechanism = dp_computations.LaplaceMechanism.create_from_epsilon(
            epsilon=2.0, l1_sensitivity=25)
        expected = ("Laplace mechanism:  parameter=12.5  eps=2.0  "
                    "l1_sensitivity=25.0")
        self.assertEqual(mechanism.describe(), expected)

    @parameterized.parameters(
        dict(l0_sensitivity=-2,
             linf_sensitivity=2,
             l1_sensitivity=None,
             l2_sensitivity=None,
             expected_error="L0 must be positive"),
        dict(l0_sensitivity=2,
             linf_sensitivity=-2,
             l1_sensitivity=-1,
             l2_sensitivity=None,
             expected_error="Linf must be positive"),
        dict(l0_sensitivity=None,
             linf_sensitivity=None,
             l1_sensitivity=0,
             l2_sensitivity=None,
             expected_error="L1 must be positive"),
        dict(l0_sensitivity=None,
             linf_sensitivity=None,
             l1_sensitivity=None,
             l2_sensitivity=-5,
             expected_error="L2 must be positive"),
        dict(l0_sensitivity=4,
             linf_sensitivity=None,
             l1_sensitivity=None,
             l2_sensitivity=None,
             expected_error="both set or both unset"),
        dict(l0_sensitivity=4,
             linf_sensitivity=2,
             l1_sensitivity=7,
             l2_sensitivity=None,
             expected_error="L1=7 != .*=8"),
        dict(l0_sensitivity=4,
             linf_sensitivity=5,
             l1_sensitivity=None,
             l2_sensitivity=9,
             expected_error="L2=9 != .*=10"),
    )
    def test_sensitivities_post_init_validation(self, l0_sensitivity,
                                                linf_sensitivity,
                                                l1_sensitivity, l2_sensitivity,
                                                expected_error):
        with self.assertRaisesRegex(ValueError, expected_error):
            dp_computations.Sensitivities(l0_sensitivity, linf_sensitivity,
                                          l1_sensitivity, l2_sensitivity)

    def test_sensitivities_post_init_l1_l2_computation(self):
        sensitivities = dp_computations.Sensitivities(l0=4, linf=5)
        self.assertEqual(sensitivities.l1, 20)
        self.assertEqual(sensitivities.l2, 10)

    @parameterized.parameters(
        dict(epsilon=2,
             l0_sensitivity=None,
             linf_sensitivity=None,
             l1_sensitivity=5,
             expected_noise_parameter=2.5),
        dict(epsilon=0.1,
             l0_sensitivity=None,
             linf_sensitivity=None,
             l1_sensitivity=3,
             expected_noise_parameter=30),
        dict(epsilon=0.5,
             l0_sensitivity=8,
             linf_sensitivity=3,
             l1_sensitivity=None,
             expected_noise_parameter=48),
    )
    def test_create_additive_mechanism_laplace(self, epsilon, l0_sensitivity,
                                               linf_sensitivity, l1_sensitivity,
                                               expected_noise_parameter):
        spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.LAPLACE)
        spec.set_eps_delta(epsilon, delta=0)
        sensitivities = dp_computations.Sensitivities(l0=l0_sensitivity,
                                                      linf=linf_sensitivity,
                                                      l1=l1_sensitivity)

        mechanism = dp_computations.create_additive_mechanism(
            spec, sensitivities)

        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_parameter,
                               delta=1e-12)

    def test_create_additive_mechanism_laplace_from_stddev(self):
        spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.LAPLACE)
        spec.set_noise_standard_deviation(7)
        sensitivities = dp_computations.Sensitivities(l1=2)

        mechanism = dp_computations.create_additive_mechanism(
            spec, sensitivities)

        self.assertAlmostEqual(mechanism.noise_parameter,
                               14 / np.sqrt(2),
                               delta=1e-12)

    @parameterized.parameters(
        dict(epsilon=2,
             delta=1e-10,
             l2_sensitivity=10,
             expected_noise_parameter=30.2734375),
        dict(epsilon=0.1,
             delta=1e-15,
             l2_sensitivity=3,
             expected_noise_parameter=213.9375),
    )
    def test_create_gaussian_mechanism(self, epsilon, delta, l2_sensitivity,
                                       expected_noise_parameter):
        spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.GAUSSIAN)
        spec.set_eps_delta(epsilon, delta)
        sensitivities = dp_computations.Sensitivities(l2=l2_sensitivity)

        mechanism = dp_computations.create_additive_mechanism(
            spec, sensitivities)

        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_parameter,
                               delta=1e-6)

    def test_create_additive_mechanism_gaussian_from_stddev(self):
        spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.GAUSSIAN)
        spec.set_noise_standard_deviation(9)
        sensitivities = dp_computations.Sensitivities(l2=2)

        mechanism = dp_computations.create_additive_mechanism(
            spec, sensitivities)

        self.assertEqual(mechanism.noise_parameter, 18)

    def test_compute_sensitivities_for_count(self):
        params = create_aggregate_params(max_partitions_contributed=4,
                                         max_contributions_per_partition=11)
        sensitivities = dp_computations.compute_sensitivities_for_count(params)
        self.assertEqual(sensitivities.l0, 4)
        self.assertEqual(sensitivities.linf, 11)
        self.assertEqual(sensitivities.l1, 44)
        self.assertEqual(sensitivities.l2, 22.0)

    def test_compute_sensitivities_for_count_max_contributions(self):
        params = create_aggregate_params(max_contributions=11)
        sensitivities = dp_computations.compute_sensitivities_for_count(params)
        self.assertIsNone(sensitivities.l0)
        self.assertIsNone(sensitivities.linf)
        self.assertEqual(sensitivities.l1, 11)
        self.assertEqual(sensitivities.l2, 11)

    def test_compute_sensitivities_for_privacy_id_count(self):
        params = create_aggregate_params(max_partitions_contributed=4,
                                         max_contributions_per_partition=11)
        sensitivities = dp_computations.compute_sensitivities_for_privacy_id_count(
            params)
        self.assertEqual(sensitivities.l0, 4)
        self.assertEqual(sensitivities.linf, 1)
        self.assertEqual(sensitivities.l1, 4)
        self.assertEqual(sensitivities.l2, 2.0)

    def test_compute_sensitivities_for_privacy_id_count_max_contributions(self):
        params = create_aggregate_params(max_contributions=9)
        sensitivities = dp_computations.compute_sensitivities_for_privacy_id_count(
            params)
        self.assertIsNone(sensitivities.l0)
        self.assertIsNone(sensitivities.linf)
        self.assertEqual(sensitivities.l1, 9)
        self.assertEqual(sensitivities.l2, 3)

    def test_compute_sensitivities_for_sum_min_max_values(self):
        params = create_aggregate_params(max_partitions_contributed=4,
                                         max_contributions_per_partition=11,
                                         min_value=-2,
                                         max_value=5)
        sensitivities = dp_computations.compute_sensitivities_for_sum(params)
        self.assertEqual(sensitivities.l0, 4)
        self.assertEqual(sensitivities.linf, 55)
        self.assertEqual(sensitivities.l1, 220)
        self.assertEqual(sensitivities.l2, 110.0)

    def test_compute_sensitivities_for_sum_min_max_per_partition(self):
        params = create_aggregate_params(max_partitions_contributed=4,
                                         max_contributions_per_partition=11,
                                         min_sum_per_partition=-2,
                                         max_sum_per_partition=5)
        sensitivities = dp_computations.compute_sensitivities_for_sum(params)
        self.assertEqual(sensitivities.l0, 4)
        self.assertEqual(sensitivities.linf, 5)
        self.assertEqual(sensitivities.l1, 20)
        self.assertEqual(sensitivities.l2, 10.0)

    def test_compute_sensitivities_for_sum_max_contributions(self):
        params = create_aggregate_params(max_contributions=5,
                                         min_value=-1,
                                         max_value=3)
        sensitivities = dp_computations.compute_sensitivities_for_sum(params)
        self.assertIsNone(sensitivities.l0)
        self.assertIsNone(sensitivities.linf)
        self.assertEqual(sensitivities.l1, 3 * 5)
        self.assertEqual(sensitivities.l2, 3 * 5)


class MeanMechanismTests(parameterized.TestCase):

    def create_mean_mechanism(self) -> dp_computations.MeanMechanism:
        count_spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.GAUSSIAN)
        count_spec.set_eps_delta(eps=1.0, delta=1e-5)
        count_sensitivities = dp_computations.Sensitivities(l0=4, linf=10)
        sum_spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.LAPLACE)
        sum_spec.set_eps_delta(eps=2.0, delta=1e-7)
        sum_sensitivities = dp_computations.Sensitivities(l0=3, linf=5)
        return dp_computations.create_mean_mechanism(5, count_spec,
                                                     count_sensitivities,
                                                     sum_spec,
                                                     sum_sensitivities)

    def test_compute_sensitivities_for_normalized_sum(self):
        params = create_aggregate_params(max_partitions_contributed=5,
                                         max_contributions_per_partition=7,
                                         min_value=50,
                                         max_value=100)
        expected = dp_computations.Sensitivities(l0=5, linf=7 * 25)
        self.assertEqual(
            dp_computations.compute_sensitivities_for_normalized_sum(params),
            expected)

    def test_compute_sensitivities_for_normalized_sum_max_contributions(self):
        params = create_aggregate_params(max_contributions=7,
                                         min_value=50,
                                         max_value=100)
        expected = dp_computations.Sensitivities(l1=7 * 25, l2=7 * 25)
        self.assertEqual(
            dp_computations.compute_sensitivities_for_normalized_sum(params),
            expected)

    def test_create_mean_mechanism(self):
        mechanism = self.create_mean_mechanism()
        self.assertEqual(mechanism._range_middle, 5)
        self.assertEqual(mechanism._count_mechanism.noise_kind,
                         pipeline_dp.NoiseKind.GAUSSIAN)
        self.assertEqual(mechanism._count_mechanism.sensitivity, 20)
        self.assertEqual(mechanism._sum_mechanism.noise_kind,
                         pipeline_dp.NoiseKind.LAPLACE)
        self.assertEqual(mechanism._sum_mechanism.sensitivity, 15)

    @patch("pipeline_dp.dp_computations.GaussianMechanism.add_noise")
    @patch("pipeline_dp.dp_computations.LaplaceMechanism.add_noise")
    def test_compute_mean(self, mock_sum_add_noise, mock_count_add_noise):
        mechanism = self.create_mean_mechanism()

        mock_count_add_noise.return_value = 10  # dp_count
        mock_sum_add_noise.return_value = 20  # dp_sum

        expected_mean = 7  # mid_range + dp_sum/dp_count = 5 + 20/10
        self.assertEqual(mechanism.compute_mean(count=11, normalized_sum=21),
                         (10, 70, expected_mean))
        mock_count_add_noise.assert_called_with(11)
        mock_sum_add_noise.assert_called_with(21)

    @patch("pipeline_dp.dp_computations.GaussianMechanism.add_noise")
    @patch("pipeline_dp.dp_computations.LaplaceMechanism.add_noise")
    def test_compute_mean_negative_dp_count(self, mock_sum_add_noise,
                                            mock_count_add_noise):
        mechanism = self.create_mean_mechanism()

        mock_count_add_noise.return_value = -1  # dp_count
        mock_sum_add_noise.return_value = 20  # dp_sum

        expected_mean = 25  # range_middle + dp_sum/max(dp_count, 1) = 5 + 20/1 = 25
        self.assertEqual(mechanism.compute_mean(count=11, normalized_sum=21),
                         (-1, -25, expected_mean))
        mock_count_add_noise.assert_called_with(11)
        mock_sum_add_noise.assert_called_with(21)


class ThresholdingMechanismTests(parameterized.TestCase):

    def create_thresholding_mechanism(
        self,
        mechanism_type: aggregate_params.MechanismType = aggregate_params.
        MechanismType.GAUSSIAN_THRESHOLDING,
        pre_threshold: Optional[int] = None
    ) -> dp_computations.ThresholdingMechanism:
        mechanism_spec = budget_accounting.MechanismSpec(mechanism_type)

        mechanism_spec.set_eps_delta(eps=1.5, delta=1e-5)
        sensitivities = dp_computations.Sensitivities(l0=4, linf=1)

        return dp_computations.create_thresholding_mechanism(
            mechanism_spec, sensitivities, pre_threshold)

    def test_create_thresholding_mechanism(self):
        # Arrange/Act
        mechanism_spec = budget_accounting.MechanismSpec(
            aggregate_params.MechanismType.GAUSSIAN_THRESHOLDING)
        mechanism_spec.set_eps_delta(eps=1.5, delta=1e-5)
        sensitivities = dp_computations.Sensitivities(l0=4, linf=1)
        mechanism = dp_computations.create_thresholding_mechanism(
            mechanism_spec, sensitivities, pre_threshold=None)

        # Assert
        self.assertEqual(mechanism._thresholding_strategy.epsilon, 1.5)
        self.assertEqual(mechanism._thresholding_strategy.delta, 1e-5)
        self.assertEqual(
            type(mechanism._thresholding_strategy).__name__,
            "GaussianPartitionSelectionStrategy")
        self.assertEqual(
            mechanism._thresholding_strategy.max_partitions_contributed, 4)
        self.assertAlmostEqual(mechanism._thresholding_strategy.threshold,
                               26.26941,
                               delta=1e-5)

    def test_create_thresholding_mechanism(self):
        mechanism = self.create_thresholding_mechanism(
            aggregate_params.MechanismType.LAPLACE_THRESHOLDING,
            pre_threshold=20)
        self.assertEqual(
            type(mechanism._thresholding_strategy).__name__,
            "PreThresholdingPartitionSelectionStrategy")
        self.assertEqual(
            mechanism._thresholding_strategy.max_partitions_contributed, 4)
        self.assertEqual(mechanism._pre_threshold, 20)

    def test_describe(self):
        mechanism = self.create_thresholding_mechanism(
            aggregate_params.MechanismType.LAPLACE_THRESHOLDING)
        self.assertEqual(
            mechanism.describe(),
            "Laplace Thresholding with threshold=33.5 eps=1.5 delta=1e-05")

    @patch(
        'pydp._pydp._partition_selection.LaplacePartitionSelectionStrategy.noised_value_if_should_keep'
    )
    def test_noised_value_if_should_keep(self, mock_function):
        mechanism = self.create_thresholding_mechanism(
            aggregate_params.MechanismType.LAPLACE_THRESHOLDING)
        mock_function.return_value = "output"
        output = mechanism.noised_value_if_should_keep(10)
        mock_function.assert_called_once_with(10)
        self.assertEqual(output, "output")


class ExponentialMechanismTests(unittest.TestCase):

    def test_one_parameter_that_has_much_greater_score_than_the_others_is_always_returned(
            self):

        class SingleValueScoringFunction(
                dp_computations.ExponentialMechanism.ScoringFunction):

            def score(self, k) -> float:
                # e^(40 / 2) = 4e9
                # e^(-40 / 2) = 2.06e-9
                return 40 if k == 0 else -40

            @property
            def global_sensitivity(self) -> float:
                return 1

            @property
            def is_monotonic(self) -> bool:
                return False

        exponential_mechanism = dp_computations.ExponentialMechanism(
            SingleValueScoringFunction())

        # weights should be: e^(40 / 2), e^(-40 / 2), e^(-40 / 2)
        # probs should be: ~1, 4.2e-18, 4.2e-18
        chosen_value = exponential_mechanism.apply(
            eps=1, inputs_to_score_col=[0, 1, 2])

        self.assertEqual(chosen_value, 0)

    def test_monotonic_single_value_scoring_function_returns_last_number(self):

        class SingleValueScoringFunction(
                dp_computations.ExponentialMechanism.ScoringFunction):

            def score(self, k) -> float:
                # e^(20) = 4e9
                # e^(-20) = 2.06e-9
                return 20 if k == 2 else -20

            @property
            def global_sensitivity(self) -> float:
                return 1

            @property
            def is_monotonic(self) -> bool:
                return True

        exponential_mechanism = dp_computations.ExponentialMechanism(
            SingleValueScoringFunction())

        # weights should be: e^(20), e^(-20), e^(-20)
        # probs should be: ~1, 4.2e-18, 4.2e-18
        chosen_value = exponential_mechanism.apply(
            eps=1, inputs_to_score_col=[0, 1, 2])

        self.assertEqual(chosen_value, 2)

    def test_constant_scoring_function_returns_all_elements_after_many_runs(
            self):

        class ConstantScoringFunction(
                dp_computations.ExponentialMechanism.ScoringFunction):

            def score(self, k) -> float:
                return 1

            @property
            def global_sensitivity(self) -> float:
                return 1

            @property
            def is_monotonic(self) -> bool:
                return True

        exponential_mechanism = dp_computations.ExponentialMechanism(
            ConstantScoringFunction())

        # probability of only 1 value is (0.5)^100 = 7.9e-31
        chosen_values = {
            exponential_mechanism.apply(eps=1, inputs_to_score_col=[0, 1])
            for _ in range(100)
        }

        self.assertSetEqual(chosen_values, {0, 1})


if __name__ == '__main__':
    absltest.main()
