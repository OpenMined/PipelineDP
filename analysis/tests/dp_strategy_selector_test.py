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
from typing import Optional

from analysis import dp_strategy_selector
import pipeline_dp
from pipeline_dp import dp_computations


class DPStrategySelectorTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="",
             epsilon=1,
             delta=1e-10,
             metric=pipeline_dp.Metrics.COUNT,
             is_public_partitions=True,
             pre_threshold=None,
             l0s=[1, 10, 10, 20, 100],
             linfs=[1, 1, 10, 1, 5],
             expected_noise_kinds=[pipeline_dp.NoiseKind.LAPLACE] * 3 +
             [pipeline_dp.NoiseKind.GAUSSIAN] * 2,
             expected_partition_selection_strategyies=[None] * 5,
             expected_post_aggregation_thresholdings=[False] * 5),)
    def test_selection_for_public_partitions(
            self, epsilon: float, delta: float, metric: pipeline_dp.Metric,
            is_public_partitions: bool, pre_threshold: Optional[int], l0s,
            linfs, expected_noise_kinds,
            expected_partition_selection_strategyies,
            expected_post_aggregation_thresholdings):
        selector = dp_strategy_selector.DPStrategySelector(
            epsilon, delta, metric, is_public_partitions, pre_threshold)
        for i in range(len(l0s)):
            sensitivities = dp_computations.Sensitivities(l0s[i], linfs[i])
            output = selector.get_dp_strategy(sensitivities)
            self.assertEqual(output.noise_kind, expected_noise_kinds[i])
            self.assertEqual(output.partition_selection_strategy,
                             expected_partition_selection_strategyies[i])
            self.assertEqual(output.post_aggregation_thresholding,
                             expected_post_aggregation_thresholdings[i])


if __name__ == '__main__':
    absltest.main()
