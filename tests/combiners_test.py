import unittest.mock as mock

from absl.testing import absltest
from absl.testing import parameterized
import pipeline_dp
import pipeline_dp.combiners as dp_combiners
import pipeline_dp.budget_accounting as ba

import numpy as np


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


class CreateCompoundCombinersTest(parameterized.TestCase):

    def _create_aggregate_params(self, metrics: list):
        return pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=metrics,
            max_partitions_contributed=0,
            max_contributions_per_partition=0,
            budget_weight=10.0)

    @parameterized.named_parameters(
        dict(testcase_name='count',
             metrics=[pipeline_dp.Metrics.COUNT],
             expected_combiner_types=[dp_combiners.CountCombiner]),
        dict(testcase_name='sum',
             metrics=[pipeline_dp.Metrics.SUM],
             expected_combiner_types=[dp_combiners.SumCombiner]),
        dict(testcase_name='privacy_id_count',
             metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
             expected_combiner_types=[dp_combiners.PrivacyIdCountCombiner]),
        dict(testcase_name='count, sum, privacy_id_count',
             metrics=[
                 pipeline_dp.Metrics.SUM, pipeline_dp.Metrics.COUNT,
                 pipeline_dp.Metrics.PRIVACY_ID_COUNT
             ],
             expected_combiner_types=[
                 dp_combiners.CountCombiner, dp_combiners.SumCombiner,
                 dp_combiners.PrivacyIdCountCombiner
             ]),
    )
    def test_create_compound_combiner(self, metrics, expected_combiner_types):
        # Arrange.
        aggregate_params = self._create_aggregate_params(metrics)

        # Mock budget accountant.
        budget_accountant = mock.Mock()
        mock_budgets = [
            f"budget{i}" for i in range(len(expected_combiner_types))
        ]
        budget_accountant.request_budget = mock.Mock(side_effect=mock_budgets)

        # Act.
        compound_combiner = dp_combiners.create_compound_combiners(
            aggregate_params, budget_accountant)

        # Assert
        budget_accountant.request_budget.assert_called_with(
            pipeline_dp.aggregate_params.MechanismType.GAUSSIAN,
            weight=aggregate_params.budget_weight)
        # Check correctness of intenal combiners
        combiners = compound_combiner._combiners
        self.assertLen(combiners, len(expected_combiner_types))
        for combiner, expect_type, expected_budget in zip(
                combiners, expected_combiner_types, mock_budgets):
            self.assertIsInstance(combiner, expect_type)
            self.assertEqual(combiner._params._mechanism_spec, expected_budget)


class CountCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.CountCombiner(params)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(2, combiner.create_accumulator([1, 2]))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
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
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        self.assertAlmostEqual(accumulator,
                               np.mean(noisified_values),
                               delta=0.5)
        self.assertTrue(
            np.var(noisified_values) > 1)  # check that noise is added


class PrivacyIdCountCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.PrivacyIdCountCombiner(params)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(1, combiner.create_accumulator([1, 2]))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
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
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        self.assertAlmostEqual(accumulator,
                               np.mean(noisified_values),
                               delta=0.5)
        self.assertTrue(
            np.var(noisified_values) > 1)  # check that noise is added


class SumCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.SumCombiner(params)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(2, combiner.create_accumulator([1, 1]))
        # Bounding on values.
        self.assertEqual(2, combiner.create_accumulator([1, 3]))
        self.assertEqual(1, combiner.create_accumulator([0, 3]))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
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
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        self.assertAlmostEqual(accumulator,
                               np.mean(noisified_values),
                               delta=0.5)
        self.assertTrue(
            np.var(noisified_values) > 1)  # check that noise is added


class CompoundCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.CompoundCombiner([
            dp_combiners.CountCombiner(params),
            dp_combiners.SumCombiner(params)
        ])

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual((1, (2, 2)), combiner.create_accumulator((1, 1)))
        self.assertEqual((1, (2, 2)), combiner.create_accumulator((1, 1)))
        self.assertEqual((1, (3, 2)), combiner.create_accumulator((0, 3, 4)))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual((2, (4, 4)),
                         combiner.merge_accumulators((1, (2, 2)), (1, (2, 2))))
        self.assertEqual((3, (4, 5)),
                         combiner.merge_accumulators((2, (2, 3)), (1, (2, 2))))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        self.assertAlmostEqual((3, [2, 3]),
                               combiner.compute_metrics((3, [2, 3])),
                               delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = (2, (2, 3))
        noisified_values = [
            combiner.compute_metrics(accumulator) for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        id_counts = []
        noised_count = []
        noised_sum = []
        for id_count, accumulators in noisified_values:
            id_counts.append(id_count)
            noised_count.append(accumulators[0])
            noised_sum.append(accumulators[1])

        self.assertTrue(all(id_count == 2 for id_count in id_counts))
        self.assertAlmostEqual(accumulator[1][0],
                               np.mean(noised_count),
                               delta=0.5)
        self.assertAlmostEqual(accumulator[1][1],
                               np.mean(noised_sum),
                               delta=0.5)
        self.assertTrue(np.var(noised_count) > 1)  # check that noise is added
        self.assertTrue(np.var(noised_sum) > 1)  # check that noise is added


if __name__ == '__main__':
    absltest.main()
