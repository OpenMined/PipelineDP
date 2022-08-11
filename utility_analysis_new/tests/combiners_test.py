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

from absl.testing import parameterized

import pipeline_dp
from utility_analysis_new import combiners


def _create_combiner_params() -> pipeline_dp.combiners.CombinerParams:
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
             params=_create_combiner_params(),
             expected_metrics=combiners.CountUtilityAnalysisMetrics(
                 count=0,
                 per_partition_contribution_error=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='one_partition_zero_error',
             num_partitions=1,
             contribution_values=(1, 2),
             params=_create_combiner_params(),
             expected_metrics=combiners.CountUtilityAnalysisMetrics(
                 count=2,
                 per_partition_contribution_error=0,
                 expected_cross_partition_error=0,
                 std_cross_partition_error=0,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
        dict(testcase_name='4_partitions_4_contributions_keep_half',
             num_partitions=4,
             contribution_values=(1, 2, 3, 4),
             params=_create_combiner_params(),
             expected_metrics=combiners.CountUtilityAnalysisMetrics(
                 count=4,
                 per_partition_contribution_error=-2,
                 expected_cross_partition_error=-1.5,
                 std_cross_partition_error=0.8660254037844386,
                 std_noise=7.46484375,
                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))
    def test_compute_metrics(self, num_partitions, contribution_values, params,
                             expected_metrics):
        utility_analysis_combiner = combiners.UtilityAnalysisCountCombiner(
            params, is_public_partitions=True)
        test_acc = utility_analysis_combiner.create_accumulator(
            (contribution_values, num_partitions))
        self.assertEqual(expected_metrics,
                         utility_analysis_combiner.compute_metrics(test_acc))

    def test_merge(self):
        utility_analysis_combiner = combiners.UtilityAnalysisCountCombiner(
            _create_combiner_params(), is_public_partitions=True)
        test_acc1 = utility_analysis_combiner.create_accumulator(((2, 3, 4), 1))
        test_acc2 = utility_analysis_combiner.create_accumulator(((6, 7, 8), 5))
        merged_acc = utility_analysis_combiner.merge_accumulators(
            test_acc1, test_acc2)

        self.assertEqual(test_acc1.count + test_acc2.count, merged_acc.count)
        self.assertEqual(
            test_acc1.per_partition_contribution_error +
            test_acc2.per_partition_contribution_error,
            merged_acc.per_partition_contribution_error)
        self.assertEqual(
            test_acc1.expected_cross_partition_error +
            test_acc2.expected_cross_partition_error,
            merged_acc.expected_cross_partition_error)
        self.assertEqual(
            test_acc1.var_cross_partition_error +
            test_acc2.var_cross_partition_error,
            merged_acc.var_cross_partition_error)


class PartitionSelectionAccumulatorTest(parameterized.TestCase):
    def test_probabilities_to_moments(self):
        probabilities = [0.1, 0.5, 0.5, 0.2]
        moments = combiners._probabilities_to_moments(probabilities)
        self.assertAlmostEqual(4, moments.count)
        self.assertAlmostEqual(1.3, moments.expectation)
        self.assertAlmostEqual(0.75, moments.variance)
        self.assertAlmostEqual(0.168, moments.third_central_moment)

    def test_add_accumulators_both_probabilities(self):
        acc1 = combiners.PartitionSelectionAccumulator([0.1, 0.2])
        acc2 = combiners.PartitionSelectionAccumulator([0.3])
        acc = acc1 + acc2
        self.assertSequenceEqual([0.1, 0.2, 0.3], acc.probabities_to_contribute)
        self.assertIsNone(acc.moments)

        acc3 = combiners.PartitionSelectionAccumulator([0.5]*99)
        acc = acc1 + acc3
        self.assertIsNone(acc.probabities_to_contribute)
        self.assertEqual(101, acc.moments.count)

    def test_add_accumulators_probabilities_moments(self):
        acc1 = combiners.PartitionSelectionAccumulator([0.1, 0.2])
        moments = combiners.SumOfRandomVariablesMoments(count=10, expectation=5, variance=50, third_central_moment=1)
        acc2 = combiners.PartitionSelectionAccumulator(moments=moments)
        acc = acc1 + acc2

        self.assertIsNone(acc.probabities_to_contribute)
        self.assertEqual(12, acc.moments.count)

    def test_add_accumulators_moments(self):
        moments = combiners.SumOfRandomVariablesMoments(count=10, expectation=5, variance=50, third_central_moment=1)
        acc1 = combiners.PartitionSelectionAccumulator(moments=moments)
        acc2 = combiners.PartitionSelectionAccumulator(moments=moments)
        acc = acc1 + acc2

        self.assertIsNone(acc.probabities_to_contribute)
        self.assertEqual(20, acc.moments.count)
        self.assertEqual(10, acc.moments.expectation)
        self.assertEqual(100, acc.moments.variance)
        self.assertEqual(2, acc.moments.third_central_moment)
