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
import copy
import dataclasses

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized
from unittest.mock import patch
from typing import Tuple

import pipeline_dp
from analysis import per_partition_combiners as combiners
from analysis import metrics
from analysis.tests import common


def _create_combiner_params_for_count() -> Tuple[
    pipeline_dp.budget_accounting.MechanismSpec, pipeline_dp.AggregateParams]:
    mechanism_spec = pipeline_dp.budget_accounting.MechanismSpec(
        mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
        _eps=1,
        _delta=0.00001)
    params = pipeline_dp.AggregateParams(
        min_value=0,
        max_value=1,
        max_partitions_contributed=1,
        max_contributions_per_partition=2,
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT],
    )
    return mechanism_spec, params


def _check_none_are_np_float64(t) -> bool:
    if not isinstance(t, tuple):
        t = dataclasses.astuple(t)
    return all(not isinstance(v, np.float64) for v in t)


def _create_sparse_combiner_acc(
        data, n_partitions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates sparse accumulators from raw data."""
    counts = np.array(list(map(len, data)))
    sums = np.array(list(map(sum, data)))
    n_partitions = np.array(n_partitions)
    return (counts, sums, n_partitions)


class CountCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.COUNT,
                 sum=0.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=0.0,
                 std_l0_bounding_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.COUNT,
                 sum=2.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=0.0,
                 std_l0_bounding_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.COUNT,
                 sum=4.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=-2.0,
                 expected_l0_bounding_error=-1.5,
                 std_l0_bounding_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values,
                             expected_metrics):
        utility_analysis_combiner = combiners.CountCombiner(
            *_create_combiner_params_for_count())
        test_acc = utility_analysis_combiner.create_accumulator(
            (np.array([len(contribution_values)]), np.array([0]),
             np.array([num_partitions])))
        got_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        common.assert_dataclasses_are_equal(self, expected_metrics, got_metrics)
        self.assertTrue(_check_none_are_np_float64(got_metrics))

    def test_merge(self):
        utility_analysis_combiner = combiners.CountCombiner(
            *_create_combiner_params_for_count())
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
        acc1 = self._create_accumulator(probabilities=[0.1, 0.2], moments=None)
        acc2 = self._create_accumulator(probabilities=[
            0.3,
        ], moments=None)
        acc = combiners._merge_partition_selection_accumulators(acc1, acc2)
        # Test that the result has probabilities.
        probabilities, moments = acc
        self.assertSequenceEqual([0.1, 0.2, 0.3], probabilities)
        self.assertIsNone(moments)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(acc))

    def test_merge_accumulators_both_probabilities_result_moments(self):
        acc1 = self._create_accumulator(probabilities=[0.1, 0.2], moments=None)
        acc2 = self._create_accumulator(probabilities=[0.5] * 99, moments=None)
        acc = combiners._merge_partition_selection_accumulators(acc1, acc2)
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
             expected_probability_to_keep=1.0,
             pre_threshold=None),
        dict(testcase_name='Small eps delta',
             eps=1,
             delta=1e-5,
             probabilities=[0.1] * 100,
             expected_probability_to_keep=0.3321336253750503,
             pre_threshold=None),
        dict(testcase_name='All probabilities = 1',
             eps=1,
             delta=1e-5,
             probabilities=[1] * 10,
             expected_probability_to_keep=0.12818308050524607,
             pre_threshold=None),
        dict(testcase_name='All probabilities = 1 with pre_threshold',
             eps=1,
             delta=1e-5,
             probabilities=[1] * 12,
             expected_probability_to_keep=0.12818308050524607,
             pre_threshold=3),
    )
    def test_partition_selection_accumulator_compute_probability(
            self, eps, delta, probabilities, expected_probability_to_keep,
            pre_threshold):
        acc = combiners.PartitionSelectionCalculator(probabilities)
        prob_to_keep = acc.compute_probability_to_keep(
            pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
            eps,
            delta,
            max_partitions_contributed=1,
            pre_threshold=pre_threshold)
        self.assertAlmostEqual(expected_probability_to_keep,
                               prob_to_keep,
                               delta=1e-10)

    @patch(
        'analysis.per_partition_combiners.PartitionSelectionCalculator.compute_probability_to_keep'
    )
    def test_partition_selection_combiner(self,
                                          mock_compute_probability_to_keep):
        mechanism_spec, params = _create_combiner_params_for_count()
        combiner = combiners.PartitionSelectionCombiner(mechanism_spec, params)
        sparse_acc = _create_sparse_combiner_acc(([1, 2, 3],), n_partitions=[8])
        acc = combiner.create_accumulator(sparse_acc)
        probabilities, moments = acc
        self.assertLen(probabilities, 1)
        self.assertEqual(1 / 8, probabilities[0])
        mock_compute_probability_to_keep.assert_not_called()
        combiner.compute_metrics(acc)
        mock_compute_probability_to_keep.assert_called_with(
            pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
            mechanism_spec.eps, mechanism_spec.delta, 1, None)


def _create_combiner_params_for_sum(
    min: float, max: float
) -> Tuple[pipeline_dp.budget_accounting.MechanismSpec,
           pipeline_dp.AggregateParams]:
    return (pipeline_dp.budget_accounting.MechanismSpec(
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


class SumCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=[0],
             contribution_values=[()],
             params=_create_combiner_params_for_sum(0, 1),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.SUM,
                 sum=0,
                 clipping_to_min_error=0,
                 clipping_to_max_error=0,
                 expected_l0_bounding_error=0,
                 std_l0_bounding_error=0,
                 std_noise=3.732421875,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_privacy_id_zero_partition_error',
             num_partitions=[1],
             contribution_values=[(1.1, 2.2)],
             params=_create_combiner_params_for_sum(0, 3.4),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.SUM,
                 sum=3.3,
                 clipping_to_min_error=0,
                 clipping_to_max_error=0,
                 expected_l0_bounding_error=0,
                 std_l0_bounding_error=0,
                 std_noise=12.690234375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='1_privacy_id_4_contributions_clip_max_error_half',
             num_partitions=[4],
             contribution_values=[(1.1, 2.2, 3.3, 4.4)],
             params=_create_combiner_params_for_sum(0, 5.5),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.SUM,
                 sum=11.0,
                 clipping_to_min_error=0,
                 clipping_to_max_error=-5.5,
                 expected_l0_bounding_error=-4.125,
                 std_l0_bounding_error=2.381569860407206,
                 std_noise=20.5283203125,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='1_privacy_id_4_partitions_4_contributions_clip_min',
             num_partitions=[4],
             contribution_values=[(0.1, 0.2, 0.3, 0.4)],
             params=_create_combiner_params_for_sum(2, 20),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.SUM,
                 sum=1.0,
                 clipping_to_min_error=1,
                 clipping_to_max_error=0,
                 expected_l0_bounding_error=-1.5,
                 std_l0_bounding_error=0.8660254037844386,
                 std_noise=74.6484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='2_privacy_ids',
             num_partitions=[2, 4],
             contribution_values=[(1,), (0.1, 0.2, 0.3, 0.4)],
             params=_create_combiner_params_for_sum(0, 0.5),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.SUM,
                 sum=2.0,
                 clipping_to_min_error=0,
                 clipping_to_max_error=-1.0,
                 expected_l0_bounding_error=-0.625,
                 std_l0_bounding_error=0.33071891388307384,
                 std_noise=1.8662109375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.SumCombiner(*params)
        sparse_acc = _create_sparse_combiner_acc(contribution_values,
                                                 num_partitions)
        test_acc = utility_analysis_combiner.create_accumulator(sparse_acc)
        actual_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        common.assert_dataclasses_are_equal(self, expected_metrics,
                                            actual_metrics)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(actual_metrics))

    def test_merge(self):
        utility_analysis_combiner = combiners.SumCombiner(
            *_create_combiner_params_for_sum(0, 20))
        test_acc1 = (0.125, 1.5, -2, -3.5, 1000)
        test_acc2 = (1, 0, -20, 3.5, 1)
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)
        self.assertSequenceEqual((1.125, 1.5, -22, 0, 1001), merged_acc)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(merged_acc))

    def test_create_accumulator_for_multi_columns(self):
        params = _create_combiner_params_for_sum(0, 5)
        combiner = combiners.SumCombiner(*params, i_column=1)
        data = (np.array([1, 1]), np.array([[1, 10],
                                            [2, 20]]), np.array([100, 150]))
        partition_sum, clipping_to_min_error, clipping_to_max_error, expected_l0_bounding_error, var_cross_partition_error = combiner.create_accumulator(
            data)
        self.assertEqual(partition_sum, 30)
        self.assertEqual(clipping_to_min_error, 0)
        self.assertEqual(clipping_to_max_error, -20)
        self.assertAlmostEqual(expected_l0_bounding_error,
                               -9.91666667,
                               delta=1e-8)
        self.assertAlmostEqual(var_cross_partition_error,
                               0.41305556,
                               delta=1e-8)


def _create_combiner_params_for_privacy_id_count() -> Tuple[
    pipeline_dp.budget_accounting.MechanismSpec, pipeline_dp.AggregateParams]:
    return (pipeline_dp.budget_accounting.MechanismSpec(
        mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
        _eps=1,
        _delta=0.00001),
            pipeline_dp.AggregateParams(
                max_partitions_contributed=2,
                max_contributions_per_partition=2,
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            ))


class PrivacyIdCountCombinerTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                 sum=0.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 std_noise=5.278441636123016,
                 expected_l0_bounding_error=0.0,
                 std_l0_bounding_error=0.0,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='single_contribution_keep_half',
             num_partitions=4,
             contribution_values=(2,),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                 sum=1.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=-0.5,
                 std_l0_bounding_error=0.5,
                 std_noise=5.278441636123016,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='multiple_contributions_keep_half',
             num_partitions=4,
             contribution_values=(2, 2, 2, 2),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                 sum=1.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=-0.5,
                 std_l0_bounding_error=0.5,
                 std_noise=5.278441636123016,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='multiple_contributions_keep_all_no_error',
             num_partitions=1,
             contribution_values=(2, 2),
             params=_create_combiner_params_for_privacy_id_count(),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                 sum=1.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=0.0,
                 std_l0_bounding_error=0.0,
                 std_noise=5.278441636123016,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.PrivacyIdCountCombiner(*params)
        sparse_acc = _create_sparse_combiner_acc([contribution_values],
                                                 [num_partitions])
        test_acc = utility_analysis_combiner.create_accumulator(sparse_acc)

        actual_metrics = utility_analysis_combiner.compute_metrics(test_acc)
        common.assert_dataclasses_are_equal(self, expected_metrics,
                                            actual_metrics)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(actual_metrics))

    def test_merge(self):
        utility_analysis_combiner = combiners.PrivacyIdCountCombiner(
            *_create_combiner_params_for_count())
        test_acc1 = [1, 2, 3]
        test_acc2 = [5, 10, -5]
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)
        self.assertSequenceEqual((6, 12, -2), merged_acc)

        # Test that no type is np.float64
        self.assertTrue(_check_none_are_np_float64(merged_acc))


class CompoundCombinerTest(parameterized.TestCase):

    def _create_combiner(self) -> combiners.CompoundCombiner:
        mechanism_spec, params = _create_combiner_params_for_count()
        count_combiner = combiners.CountCombiner(mechanism_spec, params)
        return combiners.CompoundCombiner([count_combiner],
                                          n_sum_aggregations=0)

    def _create_combiner_2_columns(self) -> combiners.CompoundCombiner:
        mechanism_spec, params1 = _create_combiner_params_for_sum(0, 1)
        sum_combiner1 = combiners.SumCombiner(mechanism_spec,
                                              params1,
                                              i_column=0)
        params2 = copy.deepcopy(params1)
        params2.max_sum_per_partition = 5
        sum_combiner2 = combiners.SumCombiner(mechanism_spec,
                                              params2,
                                              i_column=1)
        return combiners.CompoundCombiner([sum_combiner1, sum_combiner2],
                                          n_sum_aggregations=2)

    def test_create_accumulator_empty_data_multi_columns(self):

        sparse, dense = self._create_combiner_2_columns().create_accumulator(())
        self.assertEqual(sparse, ([], [], []))
        self.assertIsNone(dense)

    def test_create_accumulator_empty_data(self):
        sparse, dense = self._create_combiner().create_accumulator(())
        self.assertEqual(sparse, ([], [], []))
        self.assertIsNone(dense)

    def test_create_accumulator(self):
        combiner = self._create_combiner()
        data = [1, 2, 3]
        n_partitions = 500
        sparse, dense = combiner.create_accumulator(
            (len(data), sum(data), n_partitions))
        self.assertEqual(([len(data)], [sum(data)], [n_partitions]), sparse)
        self.assertIsNone(dense)

    def test_create_accumulator_2_sum_columns(self):
        combiner = self._create_combiner_2_columns()
        pre_aggregate_data = [1, [2, 3], 4]  # count, sum, n_partitions
        sparse, dense = combiner.create_accumulator(pre_aggregate_data)
        self.assertEqual(([1], [[2, 3]], [4]), sparse)
        self.assertIsNone(dense)

    def test_to_dense(self):
        combiner = self._create_combiner()
        sparse_acc = ([1, 3], [10, 20], [100, 200])
        dense = combiner._to_dense(sparse_acc)
        num_privacy_ids, (count_acc,) = dense
        self.assertEqual(2, num_privacy_ids)
        self.assertSequenceEqual((4, 0, -1.0, -2.98, 0.0298), count_acc)

    def test_to_dense_2_columns(self):
        combiner = self._create_combiner_2_columns()
        sparse_acc = ([1, 3], [(10, 20), (100, 200)], [100, 200])
        dense = combiner._to_dense(sparse_acc)
        num_privacy_ids, (sum1_acc, sum2_acc) = dense
        self.assertEqual(2, num_privacy_ids)
        self.assertSequenceEqual(
            (110, 0, -108, -1.9849999999999999, 0.014875000000000001), sum1_acc)
        self.assertSequenceEqual((220, 0, -210, -9.925, 0.371875), sum2_acc)

    def test_merge_sparse(self):
        combiner = self._create_combiner()
        sparse_acc1 = ([1], [10], [100])
        acc1 = (sparse_acc1, None)
        sparse_acc2 = ([11], [2], [300])
        acc2 = (sparse_acc2, None)
        sparse, dense = combiner.merge_accumulators(acc1, acc2)
        self.assertSequenceEqual(sparse, ([1, 11], [10, 2], [100, 300]))
        self.assertIsNone(dense)

    def test_merge_sparse_result_dense(self):
        combiner = self._create_combiner()
        sparse_acc1 = ([1, 3], [10, 20], [100, 200])
        acc1 = (sparse_acc1, None)
        sparse_acc2 = ([11], [2], [300])
        acc2 = (sparse_acc2, None)
        sparse, dense = combiner.merge_accumulators(acc1, acc2)
        self.assertIsNone(sparse)
        self.assertEqual(
            [3,
             ((15, 0, -10, -4.973333333333334, 0.04308888888888889),)], dense)

    def test_merge_dense(self):
        combiner = self._create_combiner()
        dense_count_acc1 = (0, 1, 2, 3)
        acc1 = (None, (1, (dense_count_acc1,)))
        dense_count_acc2 = (0.5, 0.5, 0, -4)
        acc2 = (None, (3, (dense_count_acc2,)))
        sparse, dense = combiner.merge_accumulators(acc1, acc2)

        self.assertIsNone(sparse)
        self.assertEqual(dense, (4, ((0.5, 1.5, 2, -1),)))

    def test_merge_mix_sparse_dense(self):
        combiner = self._create_combiner()
        sparse_acc1 = ([1], [10], [100])
        dense_count_acc1 = (0, 1, 2, 3)
        acc1 = (sparse_acc1, (1, (dense_count_acc1,)))
        dense_count_acc2 = (0.5, 0.5, 0, -4)
        sparse_acc2 = ([11], [2], [300])
        acc2 = (sparse_acc2, (3, (dense_count_acc2,)))
        sparse, dense = combiner.merge_accumulators(acc1, acc2)

        self.assertEqual(sparse, ([1, 11], [10, 2], [100, 300]))
        self.assertEqual(dense, (4, ((0.5, 1.5, 2, -1),)))

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             num_partitions=0,
             contribution_values=(),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.COUNT,
                 sum=0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=0.0,
                 std_l0_bounding_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.COUNT,
                 sum=2.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=0.0,
                 expected_l0_bounding_error=0.0,
                 std_l0_bounding_error=0.0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             expected_metrics=metrics.SumMetrics(
                 aggregation=pipeline_dp.Metrics.COUNT,
                 sum=4.0,
                 clipping_to_min_error=0.0,
                 clipping_to_max_error=-2.0,
                 expected_l0_bounding_error=-1.5,
                 std_l0_bounding_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values,
                             expected_metrics):
        combiner = self._create_combiner()
        acc = combiner.create_accumulator(
            (len(contribution_values), sum(contribution_values),
             num_partitions))
        self.assertEqual(expected_metrics, combiner.compute_metrics(acc)[0])

    def test_compute_metrics_mix_sparse_dense(self):
        sparse = ([1, 11], [10, 2], [100, 300])
        dense = (4, ((2, 1.5, 2, -1, 3),))
        acc = (sparse, dense)

        output = self._create_combiner().compute_metrics(acc)

        self.assertLen(output, 1)  # only 1 combiner.
        common.assert_dataclasses_are_equal(
            self,
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=14,
                               clipping_to_min_error=1.5,
                               clipping_to_max_error=-7.0,
                               expected_l0_bounding_error=-3.9833333333333334,
                               std_l0_bounding_error=1.738731977300955,
                               std_noise=7.46484375,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            output[0])

    def test_two_internal_combiners(self):
        count_mechanism_spec, count_params = _create_combiner_params_for_count()
        count_combiner = combiners.CountCombiner(count_mechanism_spec,
                                                 count_params)
        sum_mechanism_spec, sum_params = _create_combiner_params_for_sum(0, 5)
        sum_combiner = combiners.SumCombiner(sum_mechanism_spec, sum_params)
        combiner = combiners.CompoundCombiner([count_combiner, sum_combiner],
                                              n_sum_aggregations=1)

        data, n_partitions = [1, 2, 3], 100
        acc = combiner.create_accumulator((len(data), sum(data), n_partitions))

        acc = combiner.merge_accumulators(acc, acc)
        self.assertEqual((([3, 3], [6, 6], [100, 100]), None), acc)

        utility_metrics = combiner.compute_metrics(acc)
        self.assertIsInstance(utility_metrics[0], metrics.SumMetrics)
        self.assertIsInstance(utility_metrics[1], metrics.SumMetrics)


class UtilitiesTest(parameterized.TestCase):

    def test_merge_list_left_bigger(self):
        a, b = [1, 2], [3]
        result = combiners._merge_list(a, b)
        self.assertEqual(result, [1, 2, 3])
        self.assertIs(result, a)

    def test_merge_list_right_bigger(self):
        a, b = [1], [2, 3]
        result = combiners._merge_list(a, b)
        self.assertEqual(result, [2, 3, 1])
        self.assertIs(result, b)


class RawStatisticsTest(parameterized.TestCase):

    def test_create_accumulator(self):
        count, sum_, n_partitions = np.array([1,
                                              2]), np.array([1]), np.array([2])
        combiner = combiners.RawStatisticsCombiner()
        self.assertEqual(
            combiner.create_accumulator((count, sum_, n_partitions)), (2, 3))

    def test_compute_metrics(self):
        combiner = combiners.RawStatisticsCombiner()
        self.assertEqual(combiner.compute_metrics((3, 10)),
                         metrics.RawStatistics(privacy_id_count=3, count=10))


if __name__ == '__main__':
    absltest.main()
