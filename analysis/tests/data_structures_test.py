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
"""Data structures Test"""
from absl.testing import absltest
from absl.testing import parameterized

import analysis
import pipeline_dp


class MultiParameterConfiguration(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="All MultiParameterConfiguration fields unset",
             error_msg="MultiParameterConfiguration must have at least 1 "
             "non-empty attribute.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=None,
             max_sum_per_partition=None,
             noise_kind=None,
             partition_selection_strategy=None),
        dict(testcase_name="Attributes different size 1",
             error_msg="All set attributes in MultiParameterConfiguration must "
             "have the same length.",
             max_partitions_contributed=[1],
             max_contributions_per_partition=[1, 2],
             min_sum_per_partition=None,
             max_sum_per_partition=None,
             noise_kind=None,
             partition_selection_strategy=None),
        dict(testcase_name="Attributes different size 2",
             error_msg="All set attributes in MultiParameterConfiguration must "
             "have the same length.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=[1, 1, 1],
             max_sum_per_partition=[2],
             noise_kind=None,
             partition_selection_strategy=None),
        dict(testcase_name="Attributes different size 3",
             error_msg="All set attributes in MultiParameterConfiguration must "
             "have the same length.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=None,
             max_sum_per_partition=None,
             noise_kind=[pipeline_dp.NoiseKind.GAUSSIAN] * 2,
             partition_selection_strategy=[
                 pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC
             ] * 3),
        dict(testcase_name="One of min_sum_per_partition, "
             "max_sum_per_partition is None",
             error_msg="MultiParameterConfiguration: min_sum_per_partition and "
             "max_sum_per_partition must be both set or both None.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=[1, 1, 1],
             max_sum_per_partition=None,
             noise_kind=None,
             partition_selection_strategy=None),
    )
    def test_validation(self, error_msg, max_partitions_contributed,
                        max_contributions_per_partition, min_sum_per_partition,
                        max_sum_per_partition, noise_kind,
                        partition_selection_strategy):
        with self.assertRaisesRegex(ValueError, error_msg):
            analysis.MultiParameterConfiguration(
                max_partitions_contributed, max_contributions_per_partition,
                min_sum_per_partition, max_sum_per_partition, noise_kind,
                partition_selection_strategy)

    def test_get_aggregate_params(self):
        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        max_partitions_contributed = [10, 12, 15]
        noise_kind = [pipeline_dp.NoiseKind.LAPLACE] * 2 + [
            pipeline_dp.NoiseKind.GAUSSIAN
        ]
        selection_strategy = [
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
        ] * 3
        multi_params = analysis.MultiParameterConfiguration(
            max_partitions_contributed=max_partitions_contributed,
            noise_kind=noise_kind,
            partition_selection_strategy=selection_strategy)
        self.assertTrue(3, multi_params.size)

        for i in range(multi_params.size):
            ith_params = multi_params.get_aggregate_params(params, i)
            params.max_partitions_contributed = max_partitions_contributed[i]
            params.noise_kind = noise_kind[i]
            params.partition_selection_strategy = selection_strategy[i]
            self.assertEqual(params, ith_params)


if __name__ == '__main__':
    absltest.main()
