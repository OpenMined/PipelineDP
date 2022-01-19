import pipeline_dp
import pipeline_dp.combiners as combiners
import pipeline_dp.budget_accounting as ba

import numpy as np
import unittest


def _create_mechism_spec(no_noise):
    if no_noise:
        eps, delta = 1e5, 1.0 - 1e-5
    else:
        eps, delta = 10, 1e-5

    return ba.MechanismSpec(ba.MechanismType.GAUSSIAN, None, eps, delta)


def _create_aggregate_params(*metrics: pipeline_dp.Metrics):
    return pipeline_dp.AggregateParams(
        low=0,
        high=4,
        max_partitions_contributed=1,
        max_contributions_per_partition=3,
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=metrics)


class CountAccumulatorTest(unittest.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params(pipeline_dp.Metrics.COUNT)
        params = combiners.CombinerParams(mechanism_spec, aggregate_params)
        return combiners.CountCombiner(params)

    def test_create_accumulator(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual(0, combiner.create_accumulator([]))
            self.assertEqual(2, combiner.create_accumulator([1, 2]))

    def test_merge_accumulators(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual(0, combiner.merge_accumulators(0, 0))
            self.assertEqual(5, combiner.merge_accumulators(1, 4))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        self.assertAlmostEqual(3, combiner.compute_metrics(3), delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = 5
        noisy_values = [
            combiner.compute_metrics(accumulator) for _ in range(1000)
        ]
        self.assertAlmostEqual(accumulator, np.mean(noisy_values), delta=1e-1)
        self.assertTrue(np.var(noisy_values) > 1)  # check that noise is added


class MeanAccumulatorTest(unittest.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params(pipeline_dp.Metrics.MEAN)
        params = combiners.CombinerParams(mechanism_spec, aggregate_params)
        return combiners.MeanCombiner(params)

    def test_create_accumulator(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0), combiner.create_accumulator([]))
            self.assertEqual((2, 3), combiner.create_accumulator([1, 2]))

    def test_merge_accumulators(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0), combiner.merge_accumulators((0, 0),
                                                                 (0, 0)))
            self.assertEqual((5, 2), combiner.merge_accumulators((1, 0),
                                                                 (4, 2)))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        res = combiner.compute_metrics((3, 3))
        self.assertAlmostEqual(3, res.count, delta=1e-5)
        self.assertAlmostEqual(3, res.sum, delta=1e-5)
        self.assertAlmostEqual(1, res.mean, delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        count = 5
        sum = 10
        mean = 2
        noisy_values = [
            combiner.compute_metrics((count, sum)) for _ in range(1000)
        ]

        noisy_counts = [noisy_value.count for noisy_value in noisy_values]
        self.assertAlmostEqual(count, np.mean(noisy_counts), delta=5e-1)
        self.assertTrue(np.var(noisy_counts) > 1)  # check that noise is added

        noisy_sums = [noisy_value.sum for noisy_value in noisy_values]
        self.assertAlmostEqual(sum, np.mean(noisy_sums), delta=5e-1)
        self.assertTrue(np.var(noisy_sums) > 1)  # check that noise is added

        noisy_means = [noisy_value.mean for noisy_value in noisy_values]
        self.assertAlmostEqual(mean, np.mean(noisy_means), delta=5e-1)
        self.assertTrue(np.var(noisy_means) > 1)  # check that noise is added


if __name__ == '__main__':
    unittest.main()
