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
"""UtilityAnalysisCountCombinerTest."""

from absl.testing import absltest
from absl.testing import parameterized
import unittest
from unittest.mock import patch

import pipeline_dp
from utility_analysis_new import combiners


def _create_combiner_params_for_count() -> pipeline_dp.combiners.CombinerParams:
    return pipeline_dp.combiners.CombinerParams(
        pipeline_dp.budget_accounting.MechanismSpec(
            mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
            _eps=1,
            _delta=0.00001),
        pipeline_dp.AggregateParams(
            min_value=0,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=2,
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
        ))


class UtilityAnalysisCountCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             params=_create_combiner_params_for_count(),
             expected_metrics=combiners.CountMetrics(
                 count=0,
                 per_partition_error=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             params=_create_combiner_params_for_count(),
             expected_metrics=combiners.CountMetrics(
                 count=2,
                 per_partition_error=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             params=_create_combiner_params_for_count(),
             expected_metrics=combiners.CountMetrics(
                 count=4,
                 per_partition_error=-2,
                 expected_cross_partition_error=-1.5,
                 std_cross_partition_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.CountCombiner(params)
        test_acc = utility_analysis_combiner.create_accumulator(
            (len(contribution_values), 0, num_partitions))
        self.assertEqual(expected_metrics,
                         utility_analysis_combiner.compute_metrics(test_acc))

    def test_merge(self):
        utility_analysis_combiner = combiners.CountCombiner(
            _create_combiner_params_for_count())
        test_acc1 = [1, 2, 3, -4]
        test_acc2 = [5, 10, -5, 100]
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)

        self.assertSequenceEqual((6, 12, -2, 96), merged_acc)


class PartitionSelectionTest(parameterized.TestCase):

    def _create_accumulator(self, probabilities, moments):
        return (probabilities, moments)

    def test_probabilities_to_moments(self):
        probabilities = [0.1, 0.5, 0.5, 0.2]
        moments = combiners._probabilities_to_moments(probabilities)
        self.assertAlmostEqual(4, moments.count)
        self.assertAlmostEqual(1.3, moments.expectation)
        self.assertAlmostEqual(0.75, moments.variance)
        self.assertAlmostEqual(0.168, moments.third_central_moment)

    def test_merge_accumulators_both_probabilities(self):
        acc1 = self._create_accumulator(probabilities=(0.1, 0.2), moments=None)
        acc2 = self._create_accumulator(probabilities=(0.3,), moments=None)
        acc = combiners._merge_partition_selection_accumulators(acc1, acc2)
        # Test that the result has probabilities.
        probabilities, moments = acc
        self.assertSequenceEqual([0.1, 0.2, 0.3], probabilities)
        self.assertIsNone(moments)

        acc3 = self._create_accumulator(probabilities=(0.5,) * 99, moments=None)
        acc = combiners._merge_partition_selection_accumulators(acc1, acc3)
        # Test that the result has moments.
        probabilities, moments = acc
        self.assertIsNone(probabilities)
        self.assertEqual(101, moments.count)

    def test_add_accumulators_probabilities_moments(self):
        acc1 = self._create_accumulator(probabilities=(0.1, 0.2), moments=None)
        moments = combiners.SumOfRandomVariablesMoments(count=10,
                                                        expectation=5,
                                                        variance=50,
                                                        third_central_moment=1)
        acc2 = self._create_accumulator(probabilities=None, moments=moments)
        acc = combiners._merge_partition_selection_accumulators(acc1, acc2)

        # Test that the result has moments.
        probabilities, moments = acc
        self.assertIsNone(probabilities)
        self.assertEqual(12, moments.count)

    def test_add_accumulators_moments(self):
        moments = combiners.SumOfRandomVariablesMoments(count=10,
                                                        expectation=5,
                                                        variance=50,
                                                        third_central_moment=1)
        acc1 = (None, moments)
        acc2 = (None, moments)
        acc = combiners._merge_partition_selection_accumulators(acc1, acc2)

        # Test that the result has moments.
        probabilities, moments = acc
        self.assertIsNone(probabilities)
        self.assertEqual(20, moments.count)
        self.assertEqual(10, moments.expectation)
        self.assertEqual(100, moments.variance)
        self.assertEqual(2, moments.third_central_moment)

    @parameterized.named_parameters(
        dict(testcase_name='Large eps delta',
             eps=100,
             delta=0.5,
             probabilities=[1.0] * 100,
             expected_probability_to_keep=1.0),
        dict(testcase_name='Small eps delta',
             eps=1,
             delta=1e-5,
             probabilities=[0.1] * 100,
             expected_probability_to_keep=0.3321336253750503),
        dict(testcase_name='All probabilities = 1',
             eps=1,
             delta=1e-5,
             probabilities=[1] * 10,
             expected_probability_to_keep=0.12818308050524607),
    )
    def test_partition_selection_accumulator_compute_probability(
            self, eps, delta, probabilities, expected_probability_to_keep):
        acc = combiners.PartitionSelectionCalculator(probabilities)
        prob_to_keep = acc.compute_probability_to_keep(
            pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
            eps,
            delta,
            max_partitions_contributed=1)
        self.assertAlmostEqual(expected_probability_to_keep,
                               prob_to_keep,
                               delta=1e-10)

    @patch(
        'utility_analysis_new.combiners.PartitionSelectionCalculator.compute_probability_to_keep'
    )
    def test_partition_selection_combiner(self,
                                          mock_compute_probability_to_keep):
        params = _create_combiner_params_for_count()
        combiner = combiners.PartitionSelectionCombiner(params)
        data = [1, 2, 3]
        acc = combiner.create_accumulator((len(data), sum(data), 8))
        probabilities, moments = acc
        self.assertLen(probabilities, 1)
        self.assertEqual(1 / 8, probabilities[0])
        mock_compute_probability_to_keep.assert_not_called()
        combiner.compute_metrics(acc)
        mock_compute_probability_to_keep.assert_called_with(
            pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
            params.eps, params.delta, 1)


def _create_combiner_params_for_sum(
        min, max) -> pipeline_dp.combiners.CombinerParams:
    return pipeline_dp.combiners.CombinerParams(
        pipeline_dp.budget_accounting.MechanismSpec(
            mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
            _eps=1,
            _delta=0.00001),
        pipeline_dp.AggregateParams(
            max_partitions_contributed=1,
            max_contributions_per_partition=2,
            min_sum_per_partition=min,
            max_sum_per_partition=max,
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.SUM],
        ))


class UtilityAnalysisSumCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             params=_create_combiner_params_for_sum(0, 0),
             expected_metrics=combiners.SumMetrics(
                 sum=0,
                 per_partition_error_min=0,
                 per_partition_error_max=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_partition_error',
             num_partitions=1,
             contribution_values=(1.1, 2.2),
             params=_create_combiner_params_for_sum(0, 3.4),
             expected_metrics=combiners.SumMetrics(
                 sum=3.3,
                 per_partition_error_min=0,
                 per_partition_error_max=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_clip_max_error_half',
             num_partitions=4,
             contribution_values=(1.1, 2.2, 3.3, 4.4),
             params=_create_combiner_params_for_sum(0, 5.5),
             expected_metrics=combiners.SumMetrics(
                 sum=11.0,
                 per_partition_error_min=0,
                 per_partition_error_max=5.5,
                 expected_cross_partition_error=-4.125,
                 std_cross_partition_error=2.381569860407206,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_clip_min',
             num_partitions=4,
             contribution_values=(0.1, 0.2, 0.3, 0.4),
             params=_create_combiner_params_for_sum(2, 20),
             expected_metrics=combiners.SumMetrics(
                 sum=1.0,
                 per_partition_error_min=-1,
                 per_partition_error_max=0,
                 expected_cross_partition_error=-1.5,
                 std_cross_partition_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.SumCombiner(params)
        test_acc = utility_analysis_combiner.create_accumulator(
            (len(contribution_values), sum(contribution_values),
             num_partitions))
        actual_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        self.assertAlmostEqual(expected_metrics.sum, actual_metrics.sum)
        self.assertAlmostEqual(expected_metrics.per_partition_error_min,
                               actual_metrics.per_partition_error_min)
        self.assertAlmostEqual(expected_metrics.per_partition_error_max,
                               actual_metrics.per_partition_error_max)
        self.assertAlmostEqual(expected_metrics.expected_cross_partition_error,
                               actual_metrics.expected_cross_partition_error)
        self.assertAlmostEqual(expected_metrics.std_cross_partition_error,
                               actual_metrics.std_cross_partition_error)
        self.assertAlmostEqual(expected_metrics.std_noise,
                               actual_metrics.std_noise)
        self.assertEqual(expected_metrics.noise_kind, actual_metrics.noise_kind)

    def test_merge(self):
        utility_analysis_combiner = combiners.SumCombiner(
            _create_combiner_params_for_sum(0, 20))
        test_acc1 = (0.125, 1.5, -2, -3.5, 1000)
        test_acc2 = (1, 0, -20, 3.5, 1)
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)
        self.assertSequenceEqual((1.125, 1.5, -22, 0, 1001), merged_acc)


def _create_combiner_params_for_privacy_id_count(
) -> pipeline_dp.combiners.CombinerParams:
    return pipeline_dp.combiners.CombinerParams(
        pipeline_dp.budget_accounting.MechanismSpec(
            mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
            _eps=1,
            _delta=0.00001),
        pipeline_dp.AggregateParams(
            max_partitions_contributed=2,
            max_contributions_per_partition=2,
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
        ))


class UtilityAnalysisPrivacyIdCountCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=combiners.PrivacyIdCountMetrics(
                 privacy_id_count=0,
                 std_noise=10.556883272246033,
                 expected_cross_partition_error=-1,
                 std_cross_partition_error=0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='single_contribution_keep_half',
             num_partitions=4,
             contribution_values=(2,),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=combiners.PrivacyIdCountMetrics(
                 privacy_id_count=1,
                 expected_cross_partition_error=-0.5,
                 std_cross_partition_error=0.5,
                 std_noise=10.556883272246033,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='multiple_contributions_keep_half',
             num_partitions=4,
             contribution_values=(2, 2, 2, 2),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=combiners.PrivacyIdCountMetrics(
                 privacy_id_count=1,
                 expected_cross_partition_error=-0.5,
                 std_cross_partition_error=0.5,
                 std_noise=10.556883272246033,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='multiple_contributions_keep_all_no_error',
             num_partitions=1,
             contribution_values=(2, 2),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=combiners.PrivacyIdCountMetrics(
                 privacy_id_count=1,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=10.556883272246033,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.PrivacyIdCountCombiner(params)
        test_acc = utility_analysis_combiner.create_accumulator(
            (len(contribution_values), sum(contribution_values),
             num_partitions))
        actual_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        self.assertAlmostEqual(expected_metrics.privacy_id_count,
                               actual_metrics.privacy_id_count)
        self.assertAlmostEqual(expected_metrics.expected_cross_partition_error,
                               actual_metrics.expected_cross_partition_error)
        self.assertAlmostEqual(expected_metrics.std_cross_partition_error,
                               actual_metrics.std_cross_partition_error)
        self.assertAlmostEqual(expected_metrics.std_noise,
                               actual_metrics.std_noise)
        self.assertEqual(expected_metrics.noise_kind, actual_metrics.noise_kind)

    def test_merge(self):
        utility_analysis_combiner = combiners.PrivacyIdCountCombiner(
            _create_combiner_params_for_count())
        test_acc1 = [1, 2, 3]
        test_acc2 = [5, 10, -5]
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)
        self.assertSequenceEqual((6, 12, -2), merged_acc)


class UtilityAnalysisCompoundCombinerTest(parameterized.TestCase):

    def _create_combiner(self) -> combiners.CompoundCombiner:
        count_combiner = combiners.CountCombiner(
            _create_combiner_params_for_count())
        return combiners.CompoundCombiner([count_combiner],
                                          return_named_tuple=False)

    def test_create_accumulator(self):
        combiner = self._create_combiner()
        data = [1, 2, 3]
        n_partitions = 500
        sparse, dense = combiner.create_accumulator((data, n_partitions))
        self.assertEqual(sparse, [(len(data), sum(data), n_partitions)])
        self.assertIsNone(dense)

    def test_to_dense(self):
        combiner = self._create_combiner()
        sparse_acc = [(1, 10, 100), (3, 20, 200)]
        dense = combiner._to_dense(sparse_acc)
        num_privacy_ids, (count_acc,) = dense
        self.assertEqual(2, num_privacy_ids)
        self.assertSequenceEqual(count_acc, (4, -1, -2.98, 0.0298))

    def test_merge_sparse(self):
        combiner = self._create_combiner()
        sparse_acc1 = [(1, 10, 100), (3, 20, 200)]
        acc1 = (sparse_acc1, None)
        sparse_acc2 = [(11, 2, 300)]
        acc2 = (sparse_acc2, None)
        sparse, dense = combiner.merge_accumulators(acc1, acc2)
        self.assertSequenceEqual(sparse, [(1, 10, 100), (3, 20, 200),
                                          (11, 2, 300)])
        self.assertIsNone(dense)

    def test_merge_dense(self):
        combiner = self._create_combiner()
        dense_count_acc1 = (0, 1, 2, 3)
        acc1 = (None, (1, (dense_count_acc1,)))
        dense_count_acc2 = (0.5, 0.5, 0, -4)
        acc2 = (None, (3, (dense_count_acc2,)))
        sparse, dense = combiner.merge_accumulators(acc1, acc2)

        self.assertIsNone(sparse)
        self.assertEqual(dense, (4, ((0.5, 1.5, 2, -1),)))

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             expected_metrics=combiners.CountMetrics(
                 count=0,
                 per_partition_error=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             expected_metrics=combiners.CountMetrics(
                 count=2,
                 per_partition_error=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             expected_metrics=combiners.CountMetrics(
                 count=4,
                 per_partition_error=-2,
                 expected_cross_partition_error=-1.5,
                 std_cross_partition_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values,
                             expected_metrics):
        combiner = self._create_combiner()
        acc = combiner.create_accumulator((contribution_values, num_partitions))
        self.assertEqual(expected_metrics, combiner.compute_metrics(acc)[0])

    def test_two_internal_combiner(self):
        count_combiner = combiners.CountCombiner(
            _create_combiner_params_for_count())
        sum_combiner = combiners.SumCombiner(
            _create_combiner_params_for_sum(0, 5))
        combiner = combiners.CompoundCombiner([count_combiner, sum_combiner],
                                              return_named_tuple=False)

        data, n_partitions = [1, 2, 3], 100
        acc = combiner.create_accumulator((data, n_partitions))

        acc = combiner.merge_accumulators(acc, acc)

        metrics = combiner.compute_metrics(acc)
        # self.assertIsInstance(metrics[0], )


if __name__ == '__main__':
    absltest.main()
