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
"""ContributionBounders Test"""
import collections

from absl.testing import absltest
from absl.testing import parameterized

from pipeline_dp.pipeline_backend import LocalBackend
from pipeline_dp import report_generator
import analysis.contribution_bounders as contribution_bounders

MaxContributionsParams = collections.namedtuple("MaxContributionParams",
                                                ["max_contributions"])
CrossAndPerPartitionContributionParams = collections.namedtuple(
    "CrossAndPerPartitionContributionParams",
    ["max_partitions_contributed", "max_contributions_per_partition"])

# input_values is a tuple (count, sum, num_partitions_contributed)
count_aggregate_fn = lambda input_value: input_value[0]  # returns count


def _create_report_generator():
    return report_generator.ReportGenerator(None, "test")


class L0LinfContributionBounderTest(parameterized.TestCase):

    def _run_contribution_bounding(self,
                                   input,
                                   partitions_sampling_prob: float = 1.0,
                                   aggregate_fn=count_aggregate_fn):
        bounder = contribution_bounders.L0LinfAnalysisContributionBounder(
            partitions_sampling_prob)
        return list(
            bounder.bound_contributions(input, None, LocalBackend(),
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
        bound_result = self._run_contribution_bounding(input)

        expected_result = [(('pid1', 'pk2'), 2), (('pid1', 'pk1'), 2)]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_per_partition_bounding_not_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid2", 'pk2', 6)]
        bound_result = self._run_contribution_bounding(input)

        expected_result = [(('pid1', 'pk1'), 2), (('pid1', 'pk2'), 3),
                           (('pid2', 'pk2'), 1)]
        # Check per-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_cross_partition_bounding_not_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid1", 'pk3', 6),
                 ("pid1", 'pk4', 7), ("pid2", 'pk4', 8)]
        bound_result = self._run_contribution_bounding(input)

        expected_result = [(('pid1', 'pk1'), 2), (('pid1', 'pk2'), 3),
                           (('pid1', 'pk3'), 1), (('pid1', 'pk4'), 1),
                           (('pid2', 'pk4'), 1)]
        # Check per- and cross-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_cross_partition_bounding_and_sampling(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid1", 'pk3', 6),
                 ("pid1", 'pk4', 7), ("pid2", 'pk4', 8)]
        bound_result = self._run_contribution_bounding(
            input, partitions_sampling_prob=0.7)

        # 'pk1' and 'pk2' are dropped by subsampling.
        expected_result = [(('pid2', 'pk4'), 1), (('pid1', 'pk3'), 1),
                           (('pid1', 'pk4'), 1)]
        # Check per- and cross-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_cross_partition_bounding_and_2_column_values(
            self):
        input = [("pid1", 'pk1', (1, 2)), ("pid1", 'pk1', (3, 4)),
                 ("pid1", 'pk2', (-1, 0)), ("pid2", 'pk1', (5, 5))]

        bound_result = self._run_contribution_bounding(input,
                                                       aggregate_fn=lambda x: x)

        expected_result = [(('pid1', 'pk2'), (1, (-1, 0), 2, 3)),
                           (('pid1', 'pk1'), (2, (4, 6), 2, 3)),
                           (('pid2', 'pk1'), (1, (5, 5), 1, 1))]
        # Check per- and cross-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))


class LinfContributionBounderTest(parameterized.TestCase):

    def _run_contribution_bounding(self,
                                   input,
                                   partitions_sampling_prob: float = 1.0,
                                   aggregate_fn=count_aggregate_fn):
        bounder = contribution_bounders.LinfAnalysisContributionBounder(
            partitions_sampling_prob)
        return list(
            bounder.bound_contributions(input, None, LocalBackend(),
                                        _create_report_generator(),
                                        aggregate_fn))

    def test_contribution_bounding_empty_col(self):
        input = []
        max_partitions_contributed = 2
        max_contributions_per_partition = 2
        bound_result = self._run_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertEmpty(bound_result)

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4)]
        bound_result = self._run_contribution_bounding(input,
                                                       aggregate_fn=lambda x: x)
        # the output format:
        # (count_per_partition, sum_per_partition,
        # num_partition_contributed_per_privacy_id,
        # num_contribution_per_privacy_id)
        # Since no cross-partition contribution, we consider that the privacy id
        # contributes only to this partition, so
        # num_partition_contributed_per_privacy_id = 1
        # num_contribution_per_privacy_id = count_per_partition
        expected_result = [(('pid1', 'pk2'), (2, 7, 1, 2)),
                           (('pid1', 'pk1'), (2, 3, 1, 2))]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_per_partition_bounding_not_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid2", 'pk2', 6)]
        bound_result = self._run_contribution_bounding(input)

        expected_result = [(('pid1', 'pk1'), 2), (('pid1', 'pk2'), 3),
                           (('pid2', 'pk2'), 1)]
        # Check per-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_cross_partition_bounding_not_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid1", 'pk3', 6),
                 ("pid1", 'pk4', 7), ("pid2", 'pk4', 8)]

        bound_result = self._run_contribution_bounding(input)

        expected_result = [(('pid1', 'pk1'), 2), (('pid1', 'pk2'), 3),
                           (('pid1', 'pk3'), 1), (('pid1', 'pk4'), 1),
                           (('pid2', 'pk4'), 1)]
        # Check per- and cross-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_cross_partition_bounding_and_sampling(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid1", 'pk3', 6),
                 ("pid1", 'pk4', 7), ("pid2", 'pk4', 8)]

        bound_result = self._run_contribution_bounding(
            input, partitions_sampling_prob=0.7)

        # 'pk1' and 'pk2' are dropped by subsampling.
        expected_result = [(('pid2', 'pk4'), 1), (('pid1', 'pk3'), 1),
                           (('pid1', 'pk4'), 1)]
        # Check per- and cross-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_cross_partition_bounding_and_2_column_values(
            self):
        input = [("pid1", 'pk1', (1, 2)), ("pid1", 'pk1', (3, 4)),
                 ("pid1", 'pk2', (-1, 0)), ("pid2", 'pk1', (5, 5))]

        bound_result = self._run_contribution_bounding(input,
                                                       aggregate_fn=lambda x: x)

        expected_result = [(('pid1', 'pk2'), (1, (-1, 0), 1, 1)),
                           (('pid1', 'pk1'), (2, (4, 6), 1, 2)),
                           (('pid2', 'pk1'), (1, (5, 5), 1, 1))]
        # Check per- and cross-partition contribution limits are not enforced.
        self.assertEqual(set(expected_result), set(bound_result))


class NoOpContributionBounderTest(parameterized.TestCase):

    def test_contribution_bounding_doesnt_drop_contributions(self):
        # Arrange.
        # input has format (privacy_id, partition_key, (count, sum,
        # num_partitions_contributed)).
        input = [("pid1", 'pk1', (1, 2, 3)), ("pid2", 'pk2', (2, 3, 4)),
                 ("pid3", 'pk1', (10, 11, 12)),
                 ("pid3", "pk3", (100, 101, 102))]
        bounder = contribution_bounders.NoOpContributionBounder()

        # Act.
        bound_result = list(
            bounder.bound_contributions(input,
                                        params=None,
                                        backend=LocalBackend(),
                                        report_generator=None,
                                        aggregate_fn=count_aggregate_fn))

        # Assert.
        expected_result = [(("pid1", 'pk1'), 1), (("pid2", 'pk2'), 2),
                           (("pid3", 'pk1'), 10), (("pid3", 'pk3'), 100)]
        self.assertSequenceEqual(set(expected_result), set(bound_result))


if __name__ == '__main__':
    absltest.main()
