import unittest
import numpy as np
from scipy.stats import skew, kurtosis
from unittest.mock import patch
from unittest.mock import MagicMock

import pipeline_dp.dp_computations as dp_computations
from pipeline_dp.aggregate_params import NoiseKind

N_ITERATIONS = 200000


class DPComputationsTest(unittest.TestCase):

    def test_l0_sensitivity(self):
        params = dp_computations.MeanVarParams(
            eps=1,
            delta=1e-10,
            low=2,
            high=3,
            max_partitions_contributed=4,
            max_contributions_per_partition=5,
            noise_kind=NoiseKind.LAPLACE)
        self.assertEqual(params.l0_sensitivity(), 4)

    def test_l1_sensitivity(self):
        self.assertEqual(
            dp_computations.compute_l1_sensitivity(l0_sensitivity=4,
                                                   linf_sensitivity=12), 48)

    def test_l2_sensitivity(self):
        self.assertEqual(
            dp_computations.compute_l2_sensitivity(l0_sensitivity=4,
                                                   linf_sensitivity=12), 24)

    def test_compute_sigma(self):
        self.assertEqual(
            114.375,
            dp_computations.compute_sigma(eps=0.5,
                                          delta=1e-10,
                                          l2_sensitivity=10))

    def _test_laplace_noise(self, results, value, eps, l1_sensitivity):
        self.assertAlmostEqual(np.mean(results), value, delta=0.5)
        self.assertAlmostEqual(np.std(results),
                               np.sqrt(2) * l1_sensitivity / eps,
                               delta=0.5)
        self.assertAlmostEqual(skew(results), 0, delta=0.5)
        self.assertAlmostEqual(kurtosis(results), 3, delta=2)

    def _test_gaussian_noise(self, results, value, eps, delta, l2_sensitivity):
        self.assertAlmostEqual(np.mean(results), value, delta=0.5)
        self.assertAlmostEqual(np.std(results),
                               dp_computations.compute_sigma(
                                   eps, delta, l2_sensitivity),
                               delta=0.5)
        self.assertAlmostEqual(skew(results), 0, delta=0.5)
        self.assertAlmostEqual(kurtosis(results), 0, delta=0.5)

    def test_apply_laplace_mechanism(self):
        results = [
            dp_computations.apply_laplace_mechanism(value=20,
                                                    eps=0.5,
                                                    l1_sensitivity=1)
            for _ in range(1000000)
        ]
        self._test_laplace_noise(results, value=20, eps=0.5, l1_sensitivity=1)

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
        self._test_gaussian_noise(results,
                                  value=20,
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
            low=2,
            high=3,
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
        self._test_laplace_noise(results, 10, params.eps, l1_sensitivity)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        l2_sensitivity = dp_computations.compute_l2_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_count(count=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_gaussian_noise(results, 10, params.eps, params.delta,
                                  l2_sensitivity)

    def test_compute_dp_sum(self):
        params = dp_computations.MeanVarParams(
            eps=0.5,
            delta=1e-10,
            low=2,
            high=3,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)
        l0_sensitivity = params.l0_sensitivity()
        linf_sensitivity = params.max_contributions_per_partition * max(
            params.low, params.high)

        # Laplace Mechanism
        l1_sensitivity = dp_computations.compute_l1_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_sum(sum=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_laplace_noise(results, 10, params.eps, l1_sensitivity)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        l2_sensitivity = dp_computations.compute_l2_sensitivity(
            l0_sensitivity, linf_sensitivity)
        results = [
            dp_computations.compute_dp_sum(sum=10, dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        self._test_gaussian_noise(results, 10, params.eps, params.delta,
                                  l2_sensitivity)

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
            low=1,
            high=20,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.LAPLACE)

        (count_eps, count_delta), (_, _) = dp_computations.equally_split_budget(
            params.eps, params.delta, 2)
        l0_sensitivity = params.l0_sensitivity()
        count_linf_sensitivity = params.max_contributions_per_partition

        # Laplace Mechanism
        results = [
            dp_computations.compute_dp_mean(count=1000,
                                            sum=10000,
                                            dp_params=params)
            for _ in range(N_ITERATIONS)
        ]
        count_values, sum_values, mean_values = zip(*results)
        self._test_laplace_noise(
            count_values, 1000, count_eps,
            dp_computations.compute_l1_sensitivity(l0_sensitivity,
                                                   count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), 10000, delta=0.2)
        self.assertAlmostEqual(np.mean(mean_values), 10, delta=0.2)

        # Gaussian Mechanism
        params.noise_kind = NoiseKind.GAUSSIAN
        results = [
            dp_computations.compute_dp_mean(count=1000,
                                            sum=10000,
                                            dp_params=params)
            for _ in range(1500000)
        ]
        count_values, sum_values, mean_values = zip(*results)
        self._test_gaussian_noise(
            count_values, 1000, count_eps, count_delta,
            dp_computations.compute_l2_sensitivity(l0_sensitivity,
                                                   count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), 10000, delta=1)
        self.assertAlmostEqual(np.mean(mean_values), 10, delta=0.1)

    def test_compute_dp_var(self):
        params = dp_computations.MeanVarParams(
            eps=10,
            delta=1e-10,
            low=1,
            high=20,
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
            count_values, 100000, count_eps,
            dp_computations.compute_l1_sensitivity(l0_sensitivity,
                                                   count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), 1000000, delta=1)
        self.assertAlmostEqual(np.mean(sum_squares_values), 20000000, delta=1)
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
            count_values, 100000, count_eps, count_delta,
            dp_computations.compute_l2_sensitivity(l0_sensitivity,
                                                   count_linf_sensitivity))
        self.assertAlmostEqual(np.mean(sum_values), 1000000, delta=1)
        self.assertAlmostEqual(np.mean(sum_squares_values), 20000000, delta=1)
        self.assertAlmostEqual(np.mean(var_values), 100, delta=0.1)


if __name__ == '__main__':
    unittest.main()
