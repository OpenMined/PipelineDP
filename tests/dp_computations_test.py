import unittest
import numpy as np
from scipy.stats import skew, kurtosis

import pipeline_dp
from pipeline_dp.dp_computations import *


class MeanVarParams(unittest.TestCase):
    def test_l0_sensitivity(self):
        params = pipeline_dp.dp_computations.MeanVarParams(eps=1, delta=1e-10, low=2, high=3,
                                                           max_partitions_contributed=4,
                                                           max_contributions_per_partition=5,
                                                           noise_kind=pipeline_dp.NoiseKind.LAPLACE)
        self.assertEqual(params.l0_sensitivity(), 4)

    def test_l1_sensitivity(self):
        self.assertEqual(pipeline_dp.dp_computations.compute_l1_sensitivity(l0_sensitivity=4,
                                                                            linf_sensitivity=12),
                         48)

    def test_l2_sensitivity(self):
        self.assertEqual(pipeline_dp.dp_computations.compute_l2_sensitivity(l0_sensitivity=4,
                                                                            linf_sensitivity=12),
                         24)

    def test_linf_sensitivity(self):
        params = pipeline_dp.dp_computations.MeanVarParams(eps=1, delta=1e-10, low=2, high=3,
                                                           max_partitions_contributed=4,
                                                           max_contributions_per_partition=5,
                                                           noise_kind=pipeline_dp.NoiseKind.LAPLACE)

        # COUNT aggregation
        self.assertEqual(params.linf_sensitivity(metric=pipeline_dp.Metrics.COUNT), 5)

        # SUM aggregation
        # Positive low, positive high
        self.assertEqual(params.linf_sensitivity(metric=pipeline_dp.Metrics.SUM), 15)

        # Negative low, positive high
        params.low = -2
        self.assertEqual(params.linf_sensitivity(metric=pipeline_dp.Metrics.SUM), 15)

        # Negative low, negative high
        params.high = -1
        self.assertEqual(params.linf_sensitivity(metric=pipeline_dp.Metrics.SUM), 10)

    def test_compute_sigma(self):
        self.assertEqual(
            pipeline_dp.dp_computations.compute_sigma(eps=1, delta=1, l2_sensitivity=10),
            np.sqrt(2 * np.log(1.25)) * 10)
        self.assertEqual(
            pipeline_dp.dp_computations.compute_sigma(eps=0.5, delta=1e-10, l2_sensitivity=10),
            np.sqrt(2 * np.log(1.25 / 1e-10)) * 20)

    def _test_laplace_noise(self, results, value, eps, l1_sensitivity):
        self.assertAlmostEqual(np.mean(results), value, delta=0.1)
        self.assertAlmostEqual(np.std(results), np.sqrt(2) * l1_sensitivity / eps, delta=0.1)
        self.assertAlmostEqual(skew(results), 0, delta=0.1)
        self.assertAlmostEqual(kurtosis(results), 3, delta=0.1)

    def _test_gaussian_noise(self, results, value, eps, delta, l2_sensitivity):
        self.assertAlmostEqual(np.mean(results), value, delta=0.1)
        self.assertAlmostEqual(np.std(results),
                               pipeline_dp.dp_computations.compute_sigma(eps, delta,
                                                                         l2_sensitivity),
                               delta=0.1)
        self.assertAlmostEqual(skew(results), 0, delta=0.1)
        self.assertAlmostEqual(kurtosis(results), 0, delta=0.1)

    def test_apply_laplace_mechanism(self):
        results = [
            pipeline_dp.dp_computations.apply_laplace_mechanism(value=20, eps=0.5, l1_sensitivity=1)
            for _ in range(1000000)]
        self._test_laplace_noise(results, value=20, eps=0.5, l1_sensitivity=1)

    def test_apply_gaussian_mechanism(self):
        results = [
            pipeline_dp.dp_computations.apply_gaussian_mechanism(value=20, eps=0.5, delta=1e-10,
                                                                 l2_sensitivity=1) for _ in
            range(1000000)]
        self._test_gaussian_noise(results, value=20, eps=0.5, delta=1e-10, l2_sensitivity=1)

    def test_compute_dp_count(self):
        params = pipeline_dp.dp_computations.MeanVarParams(eps=0.5, delta=1e-10, low=2, high=3,
                                                           max_partitions_contributed=1,
                                                           max_contributions_per_partition=1,
                                                           noise_kind=pipeline_dp.NoiseKind.LAPLACE)
        l0_sensitivity = params.l0_sensitivity()
        linf_sensitivity = params.linf_sensitivity(pipeline_dp.Metrics.COUNT)

        # Laplace Mechanism
        l1_sensitivity = pipeline_dp.dp_computations.compute_l1_sensitivity(l0_sensitivity,
                                                                            linf_sensitivity)
        results = [pipeline_dp.dp_computations.compute_dp_count(count=10, dp_params=params) for _ in
                   range(1000000)]
        self._test_laplace_noise(results, 10, params.eps, l1_sensitivity)

        # Gaussian Mechanism
        params.noise_kind = pipeline_dp.NoiseKind.GAUSSIAN
        l2_sensitivity = pipeline_dp.dp_computations.compute_l2_sensitivity(l0_sensitivity,
                                                                            linf_sensitivity)
        results = [pipeline_dp.dp_computations.compute_dp_count(count=10, dp_params=params) for _ in
                   range(1000000)]
        self._test_gaussian_noise(results, 10, params.eps, params.delta, l2_sensitivity)

    def test_compute_dp_sum(self):
        params = pipeline_dp.dp_computations.MeanVarParams(eps=0.5, delta=1e-10, low=2, high=3,
                                                           max_partitions_contributed=1,
                                                           max_contributions_per_partition=1,
                                                           noise_kind=pipeline_dp.NoiseKind.LAPLACE)
        l0_sensitivity = params.l0_sensitivity()
        linf_sensitivity = params.linf_sensitivity(pipeline_dp.Metrics.SUM)

        # Laplace Mechanism
        l1_sensitivity = pipeline_dp.dp_computations.compute_l1_sensitivity(l0_sensitivity,
                                                                            linf_sensitivity)
        results = [pipeline_dp.dp_computations.compute_dp_sum(sum=10, dp_params=params) for _ in
                   range(1000000)]
        self._test_laplace_noise(results, 10, params.eps, l1_sensitivity)

        # Gaussian Mechanism
        params.noise_kind = pipeline_dp.NoiseKind.GAUSSIAN
        l2_sensitivity = pipeline_dp.dp_computations.compute_l2_sensitivity(l0_sensitivity,
                                                                            linf_sensitivity)
        results = [pipeline_dp.dp_computations.compute_dp_sum(sum=10, dp_params=params) for _ in
                   range(1000000)]
        self._test_gaussian_noise(results, 10, params.eps, params.delta, l2_sensitivity)


if __name__ == '__main__':
    unittest.main()
