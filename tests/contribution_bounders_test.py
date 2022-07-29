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

import collections
import numpy as np
from absl.testing import parameterized

import pipeline_dp
import pipeline_dp.contribution_bounders as contribution_bounders

MaxContributionsParams = collections.namedtuple("MaxContributionParams",
                                                ["max_contributions"])
CrossAndPerPartitionContributionParams = collections.namedtuple(
    "CrossAndPerPartitionContributionParams",
    ["max_partitions_contributed", "max_contributions_per_partition"])

aggregate_fn = lambda input_values: (len(input_values), np.sum(input_values),
                                     np.sum(np.square(input_values)))


def _create_report_generator():
    return pipeline_dp.report_generator.ReportGenerator(None, "test")


class SamplingCrossAndPerPartitionContributionBounderTest(
        parameterized.TestCase):

    def _run_contribution_bounding(self, input, max_partitions_contributed,
                                   max_contributions_per_partition):
        params = CrossAndPerPartitionContributionParams(
            max_partitions_contributed, max_contributions_per_partition)

        bounder = contribution_bounders.SamplingCrossAndPerPartitionContributionBounder(
        )
        return list(
            bounder.bound_contributions(input, params,
                                        pipeline_dp.LocalBackend(),
                                        _create_report_generator(),
                                        aggregate_fn))

    def test_contribution_bounding_empty_col(self):
        input = []
        max_partitions_contributed = max_contributions_per_partition = 2
        bound_result = self._run_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertEmpty(bound_result)

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4)]
        max_partitions_contributed = max_contributions_per_partition = 2
        bound_result = self._run_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        expected_result = [(('pid1', 'pk2'), (2, 7, 25)),
                           (('pid1', 'pk1'), (2, 3, 5))]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_per_partition_bounding_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid2", 'pk2', 6)]
        max_partitions_contributed, max_contributions_per_partition = 5, 2
        bound_result = self._run_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertEqual(3, len(bound_result))
        # Check contribution limits per-partition are enforced.
        self.assertTrue(
            all(
                map(
                    lambda op_val: op_val[1][0] <=
                    max_contributions_per_partition, bound_result)))

    def test_contribution_bounding_cross_partition_bounding_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid1", 'pk3', 6),
                 ("pid1", 'pk4', 7), ("pid2", 'pk4', 8)]
        max_partitions_contributed = 3
        max_contributions_per_partition = 5

        bound_result = self._run_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertEqual(4, len(bound_result))
        # Check contributions per partitions are enforced.
        self.assertTrue(
            all(
                map(
                    lambda op_val: op_val[1][0] <=
                    max_contributions_per_partition, bound_result)))
        # Check cross partition contributions are enforced.
        dict_of_pid_to_pk = collections.defaultdict(lambda: [])
        for key, _ in bound_result:
            dict_of_pid_to_pk[key[0]].append(key[1])
        self.assertEqual(2, len(dict_of_pid_to_pk))
        self.assertTrue(
            all(
                map(
                    lambda key: len(dict_of_pid_to_pk[key]) <=
                    max_partitions_contributed, dict_of_pid_to_pk)))


class SamplingPerPrivacyIdContributionBounderTest(parameterized.TestCase):

    def _run_contribution_bounding(self, input, max_contributions):
        params = MaxContributionsParams(max_contributions)

        bounder = contribution_bounders.SamplingPerPrivacyIdContributionBounder(
        )
        return list(
            bounder.bound_contributions(input, params,
                                        pipeline_dp.LocalBackend(),
                                        _create_report_generator(),
                                        aggregate_fn))

    def test_contribution_bounding_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk2', 2), ("pid1", 'pk3', 3),
                 ("pid1", 'pk4', 4), ("pid1", 'pk5', 5), ("pid1", 'pk6', 6)]
        max_contributions = 5
        bound_result = self._run_contribution_bounding(input, max_contributions)
        self.assertLen(bound_result, max_contributions)

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4)]
        max_contributions = 4

        bound_result = self._run_contribution_bounding(input, max_contributions)

        expected_result = [(('pid1', 'pk2'), (2, 7, 25)),
                           (('pid1', 'pk1'), (2, 3, 5))]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_empty_col(self):
        input = []
        max_contributions = 4

        bound_result = self._run_contribution_bounding(input, max_contributions)

        self.assertEmpty(bound_result)
