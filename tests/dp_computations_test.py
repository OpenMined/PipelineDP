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
import numpy as np
import typing
from scipy import stats
import math
from unittest.mock import patch
from unittest.mock import MagicMock

import pipeline_dp.dp_computations as dp_computations
from pipeline_dp.aggregate_params import NoiseKind

N_ITERATIONS = 200000
DUMMY_MIN_VALUE = 2.0
DUMMY_MAX_VALUE = 3.0


class DPComputationsTest(unittest.TestCase):

    def almost_equal(self, actual, expected, tolerance):
        return abs(expected - actual) <= tolerance

    def test_l0_sensitivity(self):
        params = dp_computations.MeanVarParams(
            eps=1,
            delta=1e-10,
            min_value=2,
            max_value=3,
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
        (statistic, pvalue) = stats.ks_2samp(results, laplace_sample)
        self.assertGreaterEqual(pvalue, 0.001)

    def _test_gaussian_kolmogorov_smirnov(self, num_trials: int, results,
                                          expected_mean: float,
                                          expected_sigma: float):
        guassian_sample = np.random.normal(expected_mean, expected_sigma,
                                           num_trials)
        (statistic, pvalue) = stats.ks_2samp(results, guassian_sample)
        self.assertGreaterEqual(pvalue, 0.001)

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

        # 99% confidence interval
        # https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
        # http://www.sjsu.edu/faculty/gerstman/EpiInfo/z-table.htm
        num_trials_double = 1.0 * num_trials
        self.assertAlmostEqual(
            num_results_within_one_std / num_trials_double,
            prob_mass_within_one_std,
            delta=2.4 * math.sqrt(prob_mass_within_one_std *
                                  (1 - prob_mass_within_one_std) / num_trials))
        self.assertAlmostEqual(
            num_results_one_to_two_stds / num_trials_double,
            prob_mass_one_to_two_stds,
            delta=2.4 * math.sqrt(prob_mass_one_to_two_stds *
                                  (1 - prob_mass_one_to_two_stds) / num_trials))
        return 0

    def _test_laplace_noise(self, num_trials: int, results,
                            expected_mean: float, l1_sensitivity: float,
                            eps: float):
        expected_beta = (l1_sensitivity / eps)
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

    def _not_all_integers(self, results: typing.Iterable[float]):
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

    def test_compute_dp_count(self):
        params = dp_computations.MeanVarParams(
            eps=0.5,
            delta=1e-10,
            min_value=0,
            max_value=0,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)
        l0_sensitivity = params.l0_sensitivity()
        linf_sensitivity = params.max_contributions_per_partition

        # Laplace Mechanism
        l1_sensitivity = dp_computations.compute_l1_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_count(count=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_laplace_noise(results=results,
                                 num_trials=N_ITERATIONS,
                                 expected_mean=10,
                                 eps=params.eps,
                                 l1_sensitivity=l1_sensitivity)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        l2_sensitivity = dp_computations.compute_l2_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_count(count=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_gaussian_noise(results=results,
                                  num_trials=N_ITERATIONS,
                                  expected_mean=10,
                                  eps=params.eps,
                                  delta=params.delta,
                                  l2_sensitivity=l2_sensitivity)

    def test_compute_dp_sum(self):
        params = dp_computations.MeanVarParams(
            eps=0.5,
            delta=1e-10,
            min_value=2,
            max_value=3,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)
        l0_sensitivity = params.l0_sensitivity()
        linf_sensitivity = params.max_contributions_per_partition * max(
            params.min_value, params.max_value)

        # Laplace Mechanism
        l1_sensitivity = dp_computations.compute_l1_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_sum(sum=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_laplace_noise(results=results,
                                 num_trials=N_ITERATIONS,
                                 expected_mean=10,
                                 eps=params.eps,
                                 l1_sensitivity=l1_sensitivity)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        l2_sensitivity = dp_computations.compute_l2_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_sum(sum=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_gaussian_noise(results=results,
                                  expected_mean=10,
                                  num_trials=N_ITERATIONS,
                                  eps=params.eps,
                                  delta=params.delta,
                                  l2_sensitivity=l2_sensitivity)

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

    def test_compute_dp_mean(self):
        params = dp_computations.MeanVarParams(
            eps=0.5,
            delta=1e-10,
            min_value=1,
            max_value=20,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)

        (count_eps, count_delta), (_, _) = dp_computations.equally_split_budget(
            params.eps, params.delta, 2)
        count_l0_sensitivity = params.l0_sensitivity()
        count_linf_sensitivity = params.max_contributions_per_partition
        count_l1_sensitivity = dp_computations.compute_l1_sensitivity(
            count_l0_sensitivity, count_linf_sensitivity)

        # Laplace Mechanism
        expected_sum = 10000
        expected_count = 1000
        results = [
            dp_computations.compute_dp_mean(count=expected_count,
                                            sum=expected_sum,
                                            dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        count_values, sum_values, mean_values = zip(*results)

        self._test_laplace_noise(results=count_values,
                                 num_trials=N_ITERATIONS,
                                 expected_mean=expected_count,
                                 eps=count_eps,
                                 l1_sensitivity=count_l1_sensitivity)
        self.assertAlmostEqual(np.mean(sum_values), expected_sum, delta=0.5)
        self.assertAlmostEqual(np.mean(mean_values),
                               expected_sum / expected_count,
                               delta=0.5)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        count_l2_sensitivity = dp_computations.compute_l2_sensitivity(
            count_l0_sensitivity, count_linf_sensitivity)
        results = [
            dp_computations.compute_dp_mean(count=expected_count,
                                            sum=expected_sum,
                                            dp_params=params)
            for _ in range(N_ITERATIONS)
        ]

        count_values, sum_values, mean_values = zip(*results)
        self._test_gaussian_noise(results=count_values,
                                  num_trials=N_ITERATIONS,
                                  expected_mean=expected_count,
                                  eps=count_eps,
                                  delta=count_delta,
                                  l2_sensitivity=count_l2_sensitivity)
        self.assertAlmostEqual(np.mean(sum_values), expected_sum, delta=1)
        self.assertAlmostEqual(np.mean(mean_values),
                               expected_sum / expected_count,
                               delta=0.1)

    def test_compute_dp_var(self):
        params = dp_computations.MeanVarParams(
            eps=10,
            delta=1e-10,
            min_value=1,
            max_value=20,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)

        (count_eps,
         count_delta), (_, _), (_, _) = dp_computations.equally_split_budget(
             params.eps, params.delta, 3)
        l0_sensitivity = params.l0_sensitivity()
        count_linf_sensitivity = params.max_contributions_per_partition

        # Laplace Mechanism
        results = [
            dp_computations.compute_dp_var(count=100000,
                                           sum=1000000,
                                           sum_squares=20000000,
                                           dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        count_values, sum_values, sum_squares_values, var_values = zip(*results)
        self._test_laplace_noise(
            results=count_values,
            num_trials=N_ITERATIONS,
            expected_mean=100000,
            eps=count_eps,
            l1_sensitivity=dp_computations.compute_l1_sensitivity(
                l0_sensitivity, count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), 1000000, delta=1)
        self.assertAlmostEqual(np.mean(sum_squares_values), 20000000, delta=2)
        self.assertAlmostEqual(np.mean(var_values), 100, delta=0.1)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        results = [
            dp_computations.compute_dp_var(count=100000,
                                           sum=1000000,
                                           sum_squares=20000000,
                                           dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        count_values, sum_values, sum_squares_values, var_values = zip(*results)

        self._test_gaussian_noise(
            results=count_values,
            num_trials=N_ITERATIONS,
            expected_mean=100000,
            eps=count_eps,
            delta=count_delta,
            l2_sensitivity=dp_computations.compute_l2_sensitivity(
                l0_sensitivity, count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), 1000000, delta=5)
        self.assertAlmostEqual(np.mean(sum_squares_values), 20000000, delta=5)
        self.assertAlmostEqual(np.mean(var_values), 100, delta=0.5)


if __name__ == '__main__':
    unittest.main()
