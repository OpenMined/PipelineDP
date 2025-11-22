# Copyright 2023 OpenMined.
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
"""Tests for the DP Strategy Selector."""

from absl.testing import absltest
from absl.testing import parameterized
from typing import List
from analysis import dp_strategy_selector
from pipeline_dp import aggregate_params
from pipeline_dp import dp_computations


class DPStrategySelectorTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="count",
             epsilon=1,
             delta=1e-10,
             metrics=[aggregate_params.Metrics.COUNT],
             l0s=[1, 10, 10, 20, 100],
             linfs=[1, 1, 10, 1, 5],
             expected_noise_kinds=[aggregate_params.NoiseKind.LAPLACE] * 3 +
             [aggregate_params.NoiseKind.GAUSSIAN] * 2),
        dict(testcase_name="sum",
             epsilon=0.1,
             delta=1e-5,
             metrics=[aggregate_params.Metrics.SUM],
             l0s=[1, 2, 3, 6],
             linfs=[1, 2, 1, 1],
             expected_noise_kinds=[aggregate_params.NoiseKind.LAPLACE] * 3 +
             [aggregate_params.NoiseKind.GAUSSIAN]),
        dict(testcase_name="count_privacy_id_count",
             epsilon=0.1,
             delta=1e-5,
             metrics=[
                 aggregate_params.Metrics.PRIVACY_ID_COUNT,
                 aggregate_params.Metrics.SUM
             ],
             l0s=[1, 2, 3, 6],
             linfs=[1, 1, 1, 1],
             expected_noise_kinds=[aggregate_params.NoiseKind.LAPLACE] * 2 +
             [aggregate_params.NoiseKind.GAUSSIAN] * 2),
    )
    def test_selection_for_public_partitions(
            self, epsilon: float, delta: float,
            metrics: aggregate_params.Metric, l0s: List[int], linfs: List[int],
            expected_noise_kinds: List[aggregate_params.NoiseKind]):
        selector = dp_strategy_selector.DPStrategySelector(
            epsilon, delta, metrics, is_public_partitions=True)
        for i in range(len(l0s)):
            sensitivities = dp_computations.Sensitivities(l0s[i], linfs[i])
            output = selector.get_dp_strategy(sensitivities)
            self.assertEqual(output.noise_kind, expected_noise_kinds[i])
            self.assertIsNone(output.partition_selection_strategy)
            self.assertFalse(output.post_aggregation_thresholding)

    @parameterized.named_parameters(
        dict(
            testcase_name="count",
            epsilon=1,
            delta=1e-8,
            metrics=[aggregate_params.Metrics.COUNT],
            l0s=[1, 2, 3, 20, 100],
            linfs=[1, 1, 10, 1, 5],
            expected_noise_kinds=[aggregate_params.NoiseKind.LAPLACE] * 3 +
            [aggregate_params.NoiseKind.GAUSSIAN] * 2,
            expected_partition_selection_strategies=[
                aggregate_params.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC
            ] * 2 +
            [aggregate_params.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
            ] * 3),
        dict(testcase_name="sum",
             epsilon=0.1,
             delta=1e-3,
             metrics=[aggregate_params.Metrics.COUNT],
             l0s=[1, 2, 5],
             linfs=[1, 1, 10],
             expected_noise_kinds=[aggregate_params.NoiseKind.LAPLACE] +
             [aggregate_params.NoiseKind.GAUSSIAN] * 2,
             expected_partition_selection_strategies=[
                 aggregate_params.PartitionSelectionStrategy.
                 GAUSSIAN_THRESHOLDING
             ] * 3),
    )
    def test_selection_for_private_partitions_wo_post_aggregation_thresholding(
        self, epsilon: float, delta: float, metrics: aggregate_params.Metric,
        l0s: List[int], linfs: List[int],
        expected_noise_kinds: List[aggregate_params.NoiseKind],
        expected_partition_selection_strategies: List[
            aggregate_params.PartitionSelectionStrategy]):
        selector = dp_strategy_selector.DPStrategySelector(
            epsilon, delta, metrics, is_public_partitions=False)
        for i in range(len(l0s)):
            sensitivities = dp_computations.Sensitivities(l0s[i], linfs[i])
            output = selector.get_dp_strategy(sensitivities)
            self.assertEqual(output.noise_kind, expected_noise_kinds[i])
            self.assertEqual(output.partition_selection_strategy,
                             expected_partition_selection_strategies[i])
            self.assertFalse(output.post_aggregation_thresholding)

    @parameterized.named_parameters(
        dict(testcase_name="l0_bound=1",
             l0=1,
             expected_noise_kind=aggregate_params.NoiseKind.LAPLACE,
             expected_partition_selection_strategy=aggregate_params.
             PartitionSelectionStrategy.LAPLACE_THRESHOLDING),
        dict(testcase_name="l0_bound=10",
             l0=10,
             expected_noise_kind=aggregate_params.NoiseKind.LAPLACE,
             expected_partition_selection_strategy=aggregate_params.
             PartitionSelectionStrategy.LAPLACE_THRESHOLDING),
        dict(testcase_name="l0_bound=25",
             l0=25,
             expected_noise_kind=aggregate_params.NoiseKind.GAUSSIAN,
             expected_partition_selection_strategy=aggregate_params.
             PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING),
    )
    def test_selection_with_post_aggregation_thresholding(
            self, l0, expected_noise_kind,
            expected_partition_selection_strategy):
        selector = dp_strategy_selector.DPStrategySelector(
            epsilon=2,
            delta=1e-12,
            metrics=[aggregate_params.Metrics.PRIVACY_ID_COUNT],
            is_public_partitions=False)
        sensitivities = dp_computations.Sensitivities(l0, 1)
        output = selector.get_dp_strategy(sensitivities)
        self.assertEqual(output.noise_kind, expected_noise_kind)
        self.assertEqual(output.partition_selection_strategy,
                         expected_partition_selection_strategy)
        self.assertTrue(output.post_aggregation_thresholding)


if __name__ == '__main__':
    absltest.main()
