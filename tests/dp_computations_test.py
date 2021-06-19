import unittest
import numpy as np

import pipeline_dp


class MeanVarParams(unittest.TestCase):
    def test_l0_sensitivity(self):
        params = pipeline_dp.MeanVarParams(eps=1, delta=1e-10, low=2, high=3,
                                           max_partitions_contributed=4,
                                           max_contributions_per_partition=5,
                                           noise_kind=pipeline_dp.NoiseKind.LAPLACE)
        self.assertEqual(params.l0_sensitivity(), 4)

    def test_l1_sensitivity(self):
        self.assertEqual(pipeline_dp.compute_l1_sensitivity(l0_sensitivity=4, linf_sensitivity=12),
                         48)

    def test_l2_sensitivity(self):
        self.assertEqual(pipeline_dp.compute_l2_sensitivity(l0_sensitivity=4, linf_sensitivity=12),
                         24)

    def test_linf_sensitivity(self):
        params = pipeline_dp.MeanVarParams(eps=1, delta=1e-10, low=2, high=3,
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
        self.assertEqual(pipeline_dp.compute_sigma(eps=1, delta=1, l2_sensitivity=10),
                         np.sqrt(2 * np.log(1.25)) * 10)
        self.assertEqual(pipeline_dp.compute_sigma(eps=0.5, delta=1e-10, l2_sensitivity=10),
                         np.sqrt(2 * np.log(1.25 / 1e-10)) * 20)

    def test_apply_laplace_mechanism(self):
        results = [pipeline_dp.apply_laplace_mechanism(value=20, eps=0.5, l1_sensitivity=1) for _ in
                   range(1000000)]
        self.assertAlmostEqual(np.mean(results), 20, 1)
        self.assertAlmostEqual(np.std(results), 2 * np.sqrt(2), 1)

    def test_apply_gaussian_mechanism(self):
        results = [
            pipeline_dp.apply_gaussian_mechanism(value=20, eps=0.5, delta=1e-10, l2_sensitivity=1)
            for _ in range(1000000)]
        self.assertAlmostEqual(np.mean(results), 20, 1)
        self.assertAlmostEqual(np.std(results),
                               pipeline_dp.compute_sigma(eps=0.5, delta=1e-10, l2_sensitivity=1), 1)

    def test_compute_dp_count(self):
        pass

    def test_compute_dp_sum(self):
        pass

    def test_compute_dp_mean(self):
        pass

    def test_compute_dp_var(self):
        pass


if __name__ == '__main__':
    unittest.main()
