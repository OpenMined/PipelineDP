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
import unittest.mock as mock
from unittest.mock import patch

from absl.testing import absltest
from absl.testing import parameterized
import typing
import pipeline_dp
import pipeline_dp.combiners as dp_combiners
import pipeline_dp.budget_accounting as ba

import numpy as np


def _create_mechanism_spec(no_noise):
    if no_noise:
        eps, delta = 1e5, 1.0 - 1e-5
    else:
        eps, delta = 10, 1e-5

    return ba.MechanismSpec(ba.MechanismType.GAUSSIAN, None, eps, delta)


def _create_aggregate_params(max_value: float = 1):
    return pipeline_dp.AggregateParams(
        min_value=0,
        max_value=max_value,
        max_partitions_contributed=1,
        max_contributions_per_partition=3,
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT])


class CreateCompoundCombinersTest(parameterized.TestCase):

    def _create_aggregate_params(self, metrics: typing.Optional[typing.List]):
        return pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=metrics,
            min_value=0,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
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
        dict(testcase_name='mean',
             metrics=[
                 pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
                 pipeline_dp.Metrics.MEAN
             ],
             expected_combiner_types=[dp_combiners.MeanCombiner]),
        dict(testcase_name='variance',
             metrics=[
                 pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
                 pipeline_dp.Metrics.MEAN, pipeline_dp.Metrics.VARIANCE
             ],
             expected_combiner_types=[dp_combiners.VarianceCombiner]),
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
        compound_combiner = dp_combiners.create_compound_combiner(
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

    @patch.multiple("pipeline_dp.combiners.CustomCombiner",
                    __abstractmethods__=set())  # Mock CustomCombiner
    def test_create_compound_combiner_with_custom_combiners(self):
        # Arrange.
        # Create Mock CustomCombiners.
        custom_combiners = [
            dp_combiners.CustomCombiner(),
            dp_combiners.CustomCombiner()
        ]

        # Mock request budget and metrics names functions.
        for i, combiner in enumerate(custom_combiners):
            combiner.request_budget = mock.Mock()

        aggregate_params = self._create_aggregate_params(None)

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(1, 1e-10)

        # Act
        compound_combiner = dp_combiners.create_compound_combiner_with_custom_combiners(
            aggregate_params, budget_accountant, custom_combiners)

        # Assert
        self.assertFalse(compound_combiner._return_named_tuple)
        for combiner in custom_combiners:
            combiner.request_budget.assert_called_once()


class CountCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
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
        self.assertAlmostEqual(3,
                               combiner.compute_metrics(3)['count'],
                               delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = 5
        noisy_values = [
            combiner.compute_metrics(accumulator)['count'] for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        self.assertAlmostEqual(accumulator, np.mean(noisy_values), delta=0.5)
        self.assertTrue(np.var(noisy_values) > 1)  # check that noise is added


class PrivacyIdCountCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
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
        self.assertAlmostEqual(3,
                               combiner.compute_metrics(3)['privacy_id_count'],
                               delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = 5
        noisy_values = [
            combiner.compute_metrics(accumulator)['privacy_id_count']
            for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        self.assertAlmostEqual(accumulator, np.mean(noisy_values), delta=0.5)
        self.assertTrue(np.var(noisy_values) > 1)  # check that noise is added


class SumCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
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
        self.assertAlmostEqual(3,
                               combiner.compute_metrics(3)['sum'],
                               delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = 5
        noisy_values = [
            combiner.compute_metrics(accumulator)['sum'] for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        self.assertAlmostEqual(accumulator, np.mean(noisy_values), delta=0.5)
        self.assertTrue(np.var(noisy_values) > 1)  # check that noise is added


class MeanCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params(max_value=4)
        metrics_to_compute = ['count', 'sum', 'mean']
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.MeanCombiner(params, metrics_to_compute)

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
        self.assertAlmostEqual(3, res['count'], delta=1e-5)
        self.assertAlmostEqual(3, res['sum'], delta=1e-5)
        self.assertAlmostEqual(1, res['mean'], delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        count = 5
        sum = 10
        mean = 2
        noisy_values = [
            combiner.compute_metrics((count, sum)) for _ in range(1000)
        ]

        noisy_counts = [noisy_value['count'] for noisy_value in noisy_values]
        self.assertAlmostEqual(count, np.mean(noisy_counts), delta=5e-1)
        self.assertTrue(np.var(noisy_counts) > 1)  # check that noise is added

        noisy_sums = [noisy_value['sum'] for noisy_value in noisy_values]
        self.assertAlmostEqual(sum, np.mean(noisy_sums), delta=5e-1)
        self.assertTrue(np.var(noisy_sums) > 1)  # check that noise is added

        noisy_means = [noisy_value['mean'] for noisy_value in noisy_values]
        self.assertAlmostEqual(mean, np.mean(noisy_means), delta=5e-1)
        self.assertTrue(np.var(noisy_means) > 1)  # check that noise is added


class VarianceCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params(max_value=4)
        metrics_to_compute = ['count', 'sum', 'mean', 'variance']
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.VarianceCombiner(params, metrics_to_compute)

    def test_create_accumulator(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0, 0), combiner.create_accumulator([]))
            self.assertEqual((2, 3, 5), combiner.create_accumulator([1, 2]))

    def test_merge_accumulators(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0, 0),
                             combiner.merge_accumulators((0, 0, 0), (0, 0, 0)))
            self.assertEqual((5, 2, 2),
                             combiner.merge_accumulators((1, 0, 0), (4, 2, 2)))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        # potential values: 1, 2, 2, 3
        res = combiner.compute_metrics((4, 8, 18))
        self.assertAlmostEqual(4, res['count'], delta=1e-5)
        self.assertAlmostEqual(8, res['sum'], delta=1e-5)
        self.assertAlmostEqual(2, res['mean'], delta=1e-5)
        self.assertAlmostEqual(0.5, res['variance'], delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        # potential values: 1, 1, 2, 3, 3
        count = 5
        sum = 10
        sum_of_squares = 24
        mean = 2
        variance = 0.8
        noisy_values = [
            combiner.compute_metrics((count, sum, sum_of_squares))
            for _ in range(1000)
        ]

        noisy_counts = [noisy_value['count'] for noisy_value in noisy_values]
        self.assertAlmostEqual(count, np.mean(noisy_counts), delta=5e-1)
        self.assertGreater(np.var(noisy_counts), 1)  # check that noise is added

        noisy_sums = [noisy_value['sum'] for noisy_value in noisy_values]
        self.assertAlmostEqual(sum, np.mean(noisy_sums), delta=1)
        self.assertGreater(np.var(noisy_sums), 1)  # check that noise is added

        noisy_means = [noisy_value['mean'] for noisy_value in noisy_values]
        self.assertAlmostEqual(mean, np.mean(noisy_means), delta=5e-1)
        self.assertGreater(np.var(noisy_means), 1)  # check that noise is added

        noisy_variances = [
            noisy_value['variance'] for noisy_value in noisy_values
        ]
        self.assertAlmostEqual(variance, np.mean(noisy_variances), delta=20)
        self.assertGreater(np.var(noisy_variances),
                           1)  # check that noise is added


class CompoundCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.CompoundCombiner([
            dp_combiners.CountCombiner(params),
            dp_combiners.SumCombiner(params)
        ],
                                             return_named_tuple=True)

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
        metrics_tuple = combiner.compute_metrics((3, [2, 3]))
        self.assertAlmostEqual(2, metrics_tuple.count, delta=1e-5)
        self.assertAlmostEqual(3, metrics_tuple.sum, delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = (2, (2, 3))
        noisy_values = [
            combiner.compute_metrics(accumulator) for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        noised_count = []
        noised_sum = []
        for metrics_tuple in noisy_values:
            noised_count.append(metrics_tuple.count)
            noised_sum.append(metrics_tuple.sum)

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
