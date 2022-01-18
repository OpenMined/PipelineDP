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


def _create_aggregate_params():
    return pipeline_dp.AggregateParams(
        low=0,
        high=1,
        max_partitions_contributed=1,
        max_contributions_per_partition=3,
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT])


class CountAccumulatorTest(unittest.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
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
        noisified_values = [
            combiner.compute_metrics(accumulator) for _ in range(1000)
        ]
        self.assertAlmostEqual(accumulator,
                               np.mean(noisified_values),
                               delta=1e-1)
        self.assertTrue(
            np.var(noisified_values) > 1)  # check that noise is added


if __name__ == '__main__':
    unittest.main()
