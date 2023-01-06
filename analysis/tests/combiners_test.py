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
import dataclasses

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from unittest.mock import patch

import pipeline_dp
from analysis import combiners
from analysis import metrics
from analysis.tests import common


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


def _check_none_are_np_float64(t) -> bool:
    if not isinstance(t, tuple):
        t = dataclasses.astuple(t)
    return all(not isinstance(v, np.float64) for v in t)


class UtilityAnalysisCountCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             params=_create_combiner_params_for_count(),
             expected_metrics=metrics.SumMetrics(
                 sum=0.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=0.0,
                 std_cross_partition_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             params=_create_combiner_params_for_count(),
             expected_metrics=metrics.SumMetrics(
                 sum=2.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=0.0,
                 std_cross_partition_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             params=_create_combiner_params_for_count(),
             expected_metrics=metrics.SumMetrics(
                 sum=4.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=-2.0,
                 expected_cross_partition_error=-1.5,
                 std_cross_partition_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.CountCombiner(params)
        test_acc = utility_analysis_combiner.create_accumulator(
            (len(contribution_values), 0, num_partitions))
        got_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        common.assert_dataclasses_are_equal(self, expected_metrics, got_metrics)
        self.assertTrue(_check_none_are_np_float64(got_metrics))

    def test_merge(self):
        utility_analysis_combiner = combiners.CountCombiner(
            _create_combiner_params_for_count())
        test_acc1 = [1, 2, 3, -4]
        test_acc2 = [5, 10, -5, 100]
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)

        self.assertSequenceEqual((6, 12, -2, 96), merged_acc)
        self.assertTrue(_check_none_are_np_float64(merged_acc))


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

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc))

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

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc))

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

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc))

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
        'analysis.combiners.PartitionSelectionCalculator.compute_probability_to_keep'
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
             expected_metrics=metrics.SumMetrics(
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
             expected_metrics=metrics.SumMetrics(
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
             expected_metrics=metrics.SumMetrics(
                 sum=11.0,
                 per_partition_error_min=0,
                 per_partition_error_max=-5.5,
                 expected_cross_partition_error=-4.125,
                 std_cross_partition_error=2.381569860407206,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_clip_min',
             num_partitions=4,
             contribution_values=(0.1, 0.2, 0.3, 0.4),
             params=_create_combiner_params_for_sum(2, 20),
             expected_metrics=metrics.SumMetrics(
                 sum=1.0,
                 per_partition_error_min=1,
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
        common.assert_dataclasses_are_equal(self, expected_metrics,
                                            actual_metrics)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(actual_metrics))

    def test_merge(self):
        utility_analysis_combiner = combiners.SumCombiner(
            _create_combiner_params_for_sum(0, 20))
        test_acc1 = (0.125, 1.5, -2, -3.5, 1000)
        test_acc2 = (1, 0, -20, 3.5, 1)
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)
        self.assertSequenceEqual((1.125, 1.5, -22, 0, 1001), merged_acc)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(merged_acc))


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
             expected_metrics=metrics.SumMetrics(
                 sum=0.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 std_noise=10.556883272246033,
                 expected_cross_partition_error=0.0,
                 std_cross_partition_error=0.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='single_contribution_keep_half',
             num_partitions=4,
             contribution_values=(2,),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 sum=1.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=-0.5,
                 std_cross_partition_error=0.5,
                 std_noise=10.556883272246033,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='multiple_contributions_keep_half',
             num_partitions=4,
             contribution_values=(2, 2, 2, 2),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 sum=1.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=-0.5,
                 std_cross_partition_error=0.5,
                 std_noise=10.556883272246033,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='multiple_contributions_keep_all_no_error',
             num_partitions=1,
             contribution_values=(2, 2),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 sum=1.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=0.0,
                 std_cross_partition_error=0.0,
                 std_noise=10.556883272246033,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.PrivacyIdCountCombiner(params)
        test_acc = utility_analysis_combiner.create_accumulator(
            (len(contribution_values), sum(contribution_values),
             num_partitions))
        actual_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        common.assert_dataclasses_are_equal(self, expected_metrics,
                                            actual_metrics)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(actual_metrics))

    def test_merge(self):
        utility_analysis_combiner = combiners.PrivacyIdCountCombiner(
            _create_combiner_params_for_count())
        test_acc1 = [1, 2, 3]
        test_acc2 = [5, 10, -5]
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)
        self.assertSequenceEqual((6, 12, -2), merged_acc)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(merged_acc))


class UtilityAnalysisCompoundCombinerTest(parameterized.TestCase):

    def _create_combiner(self) -> combiners.CompoundCombiner:
        count_combiner = combiners.CountCombiner(
            _create_combiner_params_for_count())
        return combiners.CompoundCombiner([count_combiner],
                                          return_named_tuple=False)

    def test_create_accumulator_empty_data(self):
        sparse, dense = self._create_combiner().create_accumulator(())
        self.assertEqual(sparse, [(0, 0, 0)])
        self.assertIsNone(dense)

    def test_create_accumulator(self):
        combiner = self._create_combiner()
        data = [1, 2, 3]
        n_partitions = 500
        sparse, dense = combiner.create_accumulator(
            (len(data), sum(data), n_partitions))
        self.assertEqual(sparse, [(len(data), sum(data), n_partitions)])
        self.assertIsNone(dense)

    def test_to_dense(self):
        combiner = self._create_combiner()
        sparse_acc = [(1, 10, 100), (3, 20, 200)]
        dense = combiner._to_dense(sparse_acc)
        num_privacy_ids, (count_acc,) = dense
        self.assertEqual(2, num_privacy_ids)
        self.assertSequenceEqual((4, 0, -1.0, -2.98, 0.0298), count_acc)

    def test_merge_sparse(self):
        combiner = self._create_combiner()
        sparse_acc1 = [(1, 10, 100)]
        acc1 = (sparse_acc1, None)
        sparse_acc2 = [(11, 2, 300)]
        acc2 = (sparse_acc2, None)
        sparse, dense = combiner.merge_accumulators(acc1, acc2)
        self.assertSequenceEqual(sparse, [(1, 10, 100), (11, 2, 300)])
        self.assertIsNone(dense)

    def test_merge_sparse_result_dense(self):
        combiner = self._create_combiner()
        sparse_acc1 = [(1, 10, 100), (3, 20, 200)]
        acc1 = (sparse_acc1, None)
        sparse_acc2 = [(11, 2, 300)]
        acc2 = (sparse_acc2, None)
        sparse, dense = combiner.merge_accumulators(acc1, acc2)
        self.assertIsNone(sparse)
        self.assertEqual(
            (3, ((15, 0, -10, -4.973333333333334, 0.04308888888888889),)),
            dense)

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
             expected_metrics=metrics.SumMetrics(
                 sum=0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=0.0,
                 std_cross_partition_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             expected_metrics=metrics.SumMetrics(
                 sum=2.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=0.0,
                 std_cross_partition_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             expected_metrics=metrics.SumMetrics(
                 sum=4.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=-2.0,
                 expected_cross_partition_error=-1.5,
                 std_cross_partition_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values,
                             expected_metrics):
        combiner = self._create_combiner()
        acc = combiner.create_accumulator(
            (len(contribution_values), sum(contribution_values),
             num_partitions))
        self.assertEqual(expected_metrics, combiner.compute_metrics(acc)[0])

    def test_two_internal_combiners(self):
        count_combiner = combiners.CountCombiner(
            _create_combiner_params_for_count())
        sum_combiner = combiners.SumCombiner(
            _create_combiner_params_for_sum(0, 5))
        combiner = combiners.CompoundCombiner([count_combiner, sum_combiner],
                                              return_named_tuple=False)

        data, n_partitions = [1, 2, 3], 100
        acc = combiner.create_accumulator((len(data), sum(data), n_partitions))

        acc = combiner.merge_accumulators(acc, acc)
        self.assertEqual(([(3, 6, 100), (3, 6, 100)], None), acc)

        utility_metrics = combiner.compute_metrics(acc)
        self.assertIsInstance(utility_metrics[0], metrics.SumMetrics)
        self.assertIsInstance(utility_metrics[1], metrics.SumMetrics)


class AggregateErrorMetricsAccumulatorTest(parameterized.TestCase):

    def test_add(self):
        acc = combiners.AggregateErrorMetricsAccumulator(
            num_partitions=1,
            kept_partitions_expected=0.5,
            total_aggregate=5.0,
            data_dropped_l0=1.0,
            data_dropped_linf=1.0,
            data_dropped_partition_selection=1.5,
            error_l0_expected=-1.0,
            error_linf_expected=-1.0,
            error_linf_min_expected=0.0,
            error_linf_max_expected=-1.0,
            error_l0_variance=2.0,
            error_variance=3.0,
            error_quantiles=[-2.0],
            rel_error_l0_expected=-0.2,
            rel_error_linf_expected=-0.2,
            rel_error_linf_min_expected=0.0,
            rel_error_linf_max_expected=-0.2,
            rel_error_l0_variance=0.08,
            rel_error_variance=0.12,
            rel_error_quantiles=[-0.4],
            error_expected_w_dropped_partitions=-3.5,
            rel_error_expected_w_dropped_partitions=-0.7,
            noise_std=1.0,
        )
        expected = combiners.AggregateErrorMetricsAccumulator(
            num_partitions=2,
            kept_partitions_expected=1.0,
            total_aggregate=10.0,
            data_dropped_l0=2.0,
            data_dropped_linf=2.0,
            data_dropped_partition_selection=3.0,
            error_l0_expected=-2.0,
            error_linf_expected=-2.0,
            error_linf_min_expected=0.0,
            error_linf_max_expected=-2.0,
            error_l0_variance=4.0,
            error_variance=6.0,
            error_quantiles=[-4.0],
            rel_error_l0_expected=-0.4,
            rel_error_linf_expected=-0.4,
            rel_error_linf_min_expected=0.0,
            rel_error_linf_max_expected=-0.4,
            rel_error_l0_variance=0.16,
            rel_error_variance=0.24,
            rel_error_quantiles=[-0.8],
            error_expected_w_dropped_partitions=-7,
            rel_error_expected_w_dropped_partitions=-1.4,
            noise_std=1.0,
        )
        acc_sum = acc + acc
        self.assertEqual(expected, acc_sum)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc_sum))


class SumAggregateErrorMetricsCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='Sum without public partitions',
             metric_type=metrics.AggregateMetricType.SUM,
             probability_to_keep=0.5,
             metric=metrics.SumMetrics(
                 sum=5.0,
                 per_partition_error_min=1.0,
                 per_partition_error_max=-2.0,
                 expected_cross_partition_error=-2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=0.5,
                 total_aggregate=5.0,
                 data_dropped_l0=0.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-1.0,
                 error_linf_expected=-0.5,
                 error_linf_min_expected=0.5,
                 error_linf_max_expected=-1.0,
                 error_l0_variance=2.0,
                 error_variance=2.5,
                 error_quantiles=[-1.5],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.1,
                 rel_error_linf_min_expected=0.1,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.1,
                 rel_error_quantiles=[-0.3],
                 error_expected_w_dropped_partitions=-4.0,
                 rel_error_expected_w_dropped_partitions=-0.8,
                 noise_std=1.0,
             )),
        dict(testcase_name='Sum with public partitions',
             metric_type=metrics.AggregateMetricType.SUM,
             probability_to_keep=1.0,
             metric=metrics.SumMetrics(
                 sum=5.0,
                 per_partition_error_min=2.0,
                 per_partition_error_max=-1.0,
                 expected_cross_partition_error=2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=5.0,
                 data_dropped_l0=0.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=2.0,
                 error_linf_expected=1.0,
                 error_linf_min_expected=2.0,
                 error_linf_max_expected=-1.0,
                 error_l0_variance=4.0,
                 error_variance=5.0,
                 error_quantiles=[3.0],
                 rel_error_l0_expected=0.4,
                 rel_error_linf_expected=0.2,
                 rel_error_linf_min_expected=0.4,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.2,
                 rel_error_quantiles=[0.6],
                 error_expected_w_dropped_partitions=3.0,
                 rel_error_expected_w_dropped_partitions=0.6,
                 noise_std=1.0,
             )),
        dict(testcase_name='Sum with public partitions and negative sum',
             metric_type=metrics.AggregateMetricType.SUM,
             probability_to_keep=1.0,
             metric=metrics.SumMetrics(
                 sum=-5.0,
                 per_partition_error_min=2.0,
                 per_partition_error_max=-1.0,
                 expected_cross_partition_error=-2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=-5.0,
                 data_dropped_l0=0.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-2.0,
                 error_linf_expected=1.0,
                 error_linf_min_expected=2.0,
                 error_linf_max_expected=-1.0,
                 error_l0_variance=4.0,
                 error_variance=5.0,
                 error_quantiles=[-1.0],
                 rel_error_l0_expected=-0.4,
                 rel_error_linf_expected=0.2,
                 rel_error_linf_min_expected=0.4,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.2,
                 rel_error_quantiles=[-0.2],
                 error_expected_w_dropped_partitions=-1.0,
                 rel_error_expected_w_dropped_partitions=-0.2,
                 noise_std=1.0,
             )),
        dict(testcase_name='Count without public partitions',
             metric_type=metrics.AggregateMetricType.COUNT,
             probability_to_keep=0.5,
             metric=metrics.SumMetrics(
                 sum=5.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=-2.0,
                 expected_cross_partition_error=-2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=0.5,
                 total_aggregate=5.0,
                 data_dropped_l0=2.0,
                 data_dropped_linf=2.0,
                 data_dropped_partition_selection=0.5,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=-1.0,
                 error_l0_variance=2.0,
                 error_variance=2.5,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.1,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-4.5,
                 rel_error_expected_w_dropped_partitions=-0.9,
                 noise_std=1.0,
             )),
        dict(testcase_name='Count with public partitions',
             metric_type=metrics.AggregateMetricType.COUNT,
             probability_to_keep=1.0,
             metric=metrics.SumMetrics(
                 sum=5.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=-2.0,
                 expected_cross_partition_error=-2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=5.0,
                 data_dropped_l0=2.0,
                 data_dropped_linf=2.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-2.0,
                 error_linf_expected=-2.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=-2.0,
                 error_l0_variance=4.0,
                 error_variance=5.0,
                 error_quantiles=[-4.0],
                 rel_error_l0_expected=-0.4,
                 rel_error_linf_expected=-0.4,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=-0.4,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.2,
                 rel_error_quantiles=[-0.8],
                 error_expected_w_dropped_partitions=-4.0,
                 rel_error_expected_w_dropped_partitions=-0.8,
                 noise_std=1.0,
             )),
        dict(testcase_name='PrivacyIdCount without public partitions',
             metric_type=metrics.AggregateMetricType.PRIVACY_ID_COUNT,
             probability_to_keep=0.5,
             metric=metrics.SumMetrics(
                 sum=5.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=-2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=0.5,
                 total_aggregate=5.0,
                 data_dropped_l0=2.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=1.5,
                 error_l0_expected=-1.0,
                 error_linf_expected=0.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=0.0,
                 error_l0_variance=2.0,
                 error_variance=2.5,
                 error_quantiles=[-1.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=0.0,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=0.0,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.1,
                 rel_error_quantiles=[-0.2],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             )),
        dict(testcase_name='PrivacyIdCount with public partitions',
             metric_type=metrics.AggregateMetricType.PRIVACY_ID_COUNT,
             probability_to_keep=1.0,
             metric=metrics.SumMetrics(
                 sum=5.0,
                 per_partition_error_min=0.0,
                 per_partition_error_max=0.0,
                 expected_cross_partition_error=-2.0,
                 std_cross_partition_error=2.0,
                 std_noise=1.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ),
             expected=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=5.0,
                 data_dropped_l0=2.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-2.0,
                 error_linf_expected=0.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=0.0,
                 error_l0_variance=4.0,
                 error_variance=5.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.4,
                 rel_error_linf_expected=0.0,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=0.0,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.2,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-2.0,
                 rel_error_expected_w_dropped_partitions=-0.4,
                 noise_std=1.0,
             )))
    def test_create_accumulator(self, metric_type, probability_to_keep, metric,
                                expected):
        combiner = combiners.SumAggregateErrorMetricsCombiner(
            metric_type, [0.5])
        acc = combiner.create_accumulator(metric, probability_to_keep)
        common.assert_dataclasses_are_equal(self, expected, acc)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc))

    @parameterized.named_parameters(
        dict(testcase_name='Sum with public partitions',
             metric_type=metrics.AggregateMetricType.SUM,
             acc=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=0.5,
                 total_aggregate=5.0,
                 data_dropped_l0=0.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.5,
                 error_linf_max_expected=-1.5,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.1,
                 rel_error_linf_max_expected=-0.3,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             ),
             expected=metrics.AggregateErrorMetrics(
                 metric_type=metrics.AggregateMetricType.SUM,
                 ratio_data_dropped_l0=0.0,
                 ratio_data_dropped_linf=0.0,
                 ratio_data_dropped_partition_selection=0.0,
                 error_l0_expected=-2.0,
                 error_linf_expected=-2.0,
                 error_linf_min_expected=1.0,
                 error_linf_max_expected=-3.0,
                 error_expected=-4.0,
                 error_l0_variance=4.0,
                 error_variance=6.0,
                 error_quantiles=[-4.0],
                 rel_error_l0_expected=-0.4,
                 rel_error_linf_expected=-0.4,
                 rel_error_linf_min_expected=0.2,
                 rel_error_linf_max_expected=-0.6,
                 rel_error_expected=-0.8,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.24,
                 rel_error_quantiles=[-0.8],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             )),
        dict(testcase_name='Sum without public partitions',
             metric_type=metrics.AggregateMetricType.SUM,
             acc=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=5.0,
                 data_dropped_l0=0.0,
                 data_dropped_linf=0.0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.5,
                 error_linf_max_expected=-1.5,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.1,
                 rel_error_linf_max_expected=-0.3,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-2.0,
                 rel_error_expected_w_dropped_partitions=-0.4,
                 noise_std=1.0,
             ),
             expected=metrics.AggregateErrorMetrics(
                 metric_type=metrics.AggregateMetricType.SUM,
                 ratio_data_dropped_l0=0.0,
                 ratio_data_dropped_linf=0.0,
                 ratio_data_dropped_partition_selection=0.0,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.5,
                 error_linf_max_expected=-1.5,
                 error_expected=-2.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.1,
                 rel_error_linf_max_expected=-0.3,
                 rel_error_expected=-0.4,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-2.0,
                 rel_error_expected_w_dropped_partitions=-0.4,
                 noise_std=1.0,
             )),
        dict(testcase_name='Count without public partitions',
             metric_type=metrics.AggregateMetricType.COUNT,
             acc=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=0.5,
                 total_aggregate=5.0,
                 data_dropped_l0=2.0,
                 data_dropped_linf=2.0,
                 data_dropped_partition_selection=1.5,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=-1.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             ),
             expected=metrics.AggregateErrorMetrics(
                 metric_type=metrics.AggregateMetricType.COUNT,
                 ratio_data_dropped_l0=0.4,
                 ratio_data_dropped_linf=0.4,
                 ratio_data_dropped_partition_selection=0.3,
                 error_l0_expected=-2.0,
                 error_linf_expected=-2.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=-2.0,
                 error_expected=-4.0,
                 error_l0_variance=4.0,
                 error_variance=6.0,
                 error_quantiles=[-4.0],
                 rel_error_l0_expected=-0.4,
                 rel_error_linf_expected=-0.4,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=-0.4,
                 rel_error_expected=-0.8,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.24,
                 rel_error_quantiles=[-0.8],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             )),
        dict(testcase_name='Count with public partitions',
             metric_type=metrics.AggregateMetricType.COUNT,
             acc=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=5.0,
                 data_dropped_l0=1.0,
                 data_dropped_linf=1.0,
                 data_dropped_partition_selection=0,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=-1.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-2.0,
                 rel_error_expected_w_dropped_partitions=-0.4,
                 noise_std=1.0,
             ),
             expected=metrics.AggregateErrorMetrics(
                 metric_type=metrics.AggregateMetricType.COUNT,
                 ratio_data_dropped_l0=0.2,
                 ratio_data_dropped_linf=0.2,
                 ratio_data_dropped_partition_selection=0,
                 error_l0_expected=-1.0,
                 error_linf_expected=-1.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=-1.0,
                 error_expected=-2.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=-0.2,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=-0.2,
                 rel_error_expected=-0.4,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-2.0,
                 rel_error_expected_w_dropped_partitions=-0.4,
                 noise_std=1.0,
             )),
        dict(testcase_name='PrivacyIdCount without public partitions',
             metric_type=metrics.AggregateMetricType.PRIVACY_ID_COUNT,
             acc=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=0.5,
                 total_aggregate=5.0,
                 data_dropped_l0=2.0,
                 data_dropped_linf=0,
                 data_dropped_partition_selection=1.5,
                 error_l0_expected=-1.0,
                 error_linf_expected=0.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=0.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-1.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=0.0,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=0.0,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.2],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             ),
             expected=metrics.AggregateErrorMetrics(
                 metric_type=metrics.AggregateMetricType.PRIVACY_ID_COUNT,
                 ratio_data_dropped_l0=0.4,
                 ratio_data_dropped_linf=0.0,
                 ratio_data_dropped_partition_selection=0.3,
                 error_l0_expected=-2.0,
                 error_linf_expected=0.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=0.0,
                 error_expected=-2.0,
                 error_l0_variance=4.0,
                 error_variance=6.0,
                 error_quantiles=[-2.0],
                 rel_error_l0_expected=-0.4,
                 rel_error_linf_expected=0.0,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=0.0,
                 rel_error_expected=-0.4,
                 rel_error_l0_variance=0.16,
                 rel_error_variance=0.24,
                 rel_error_quantiles=[-0.4],
                 error_expected_w_dropped_partitions=-3.5,
                 rel_error_expected_w_dropped_partitions=-0.7,
                 noise_std=1.0,
             )),
        dict(testcase_name='PrivacyIdCount with public partitions',
             metric_type=metrics.AggregateMetricType.PRIVACY_ID_COUNT,
             acc=combiners.AggregateErrorMetricsAccumulator(
                 num_partitions=1,
                 kept_partitions_expected=1.0,
                 total_aggregate=5.0,
                 data_dropped_l0=1.0,
                 data_dropped_linf=0,
                 data_dropped_partition_selection=0.0,
                 error_l0_expected=-1.0,
                 error_linf_expected=0.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=0.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-1.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=0.0,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=0.0,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.2],
                 error_expected_w_dropped_partitions=-1.0,
                 rel_error_expected_w_dropped_partitions=-0.1,
                 noise_std=1.0,
             ),
             expected=metrics.AggregateErrorMetrics(
                 metric_type=metrics.AggregateMetricType.PRIVACY_ID_COUNT,
                 ratio_data_dropped_l0=0.2,
                 ratio_data_dropped_linf=0.0,
                 ratio_data_dropped_partition_selection=0.0,
                 error_l0_expected=-1.0,
                 error_linf_expected=0.0,
                 error_linf_min_expected=0.0,
                 error_linf_max_expected=0.0,
                 error_expected=-1.0,
                 error_l0_variance=2.0,
                 error_variance=3.0,
                 error_quantiles=[-1.0],
                 rel_error_l0_expected=-0.2,
                 rel_error_linf_expected=0.0,
                 rel_error_linf_min_expected=0.0,
                 rel_error_linf_max_expected=0.0,
                 rel_error_expected=-0.2,
                 rel_error_l0_variance=0.08,
                 rel_error_variance=0.12,
                 rel_error_quantiles=[-0.2],
                 error_expected_w_dropped_partitions=-1.0,
                 rel_error_expected_w_dropped_partitions=-0.1,
                 noise_std=1.0,
             )),
    )
    def test_compute_metrics(self, metric_type, acc, expected):
        combiner = combiners.SumAggregateErrorMetricsCombiner(
            metric_type, [0.5])
        metric = combiner.compute_metrics(acc)
        common.assert_dataclasses_are_equal(self, expected, metric)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc))


if __name__ == '__main__':
    absltest.main()
