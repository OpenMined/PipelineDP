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


class ContributionBoundersTest(parameterized.TestCase):

    aggregate_fn = lambda input_values: (len(input_values), np.sum(
        input_values), np.sum(np.square(input_values)))

    def _create_report_generator(self):
        return pipeline_dp.report_generator.ReportGenerator(None, "test")

    def _run_l0_linf_contribution_bounding(self, input,
                                           max_partitions_contributed,
                                           max_contributions_per_partition):
        params = CrossAndPerPartitionContributionParams(
            max_partitions_contributed, max_contributions_per_partition)

        bounder = contribution_bounders.SamplingCrossAndPerPartitionContributionBounder(
        )
        return list(
            bounder.bound_contributions(input, params,
                                        pipeline_dp.LocalBackend(),
                                        self._create_report_generator(),
                                        ContributionBoundersTest.aggregate_fn))

    def test_contribution_bounding_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk2', 2), ("pid1", 'pk3', 3),
                 ("pid1", 'pk4', 4), ("pid1", 'pk5', 5), ("pid2", 'pk1', 6)]
        params = MaxContributionsParams(4)

        bounder = contribution_bounders.SamplingPerPrivacyIdContributionBounder(
        )
        bound_result = list(
            bounder.bound_contributions(input, params,
                                        pipeline_dp.LocalBackend(),
                                        self._create_report_generator(),
                                        ContributionBoundersTest.aggregate_fn))
        self.assertLen(bound_result, 5)

    def test_contribution_bounding_empty_col(self):
        input = []
        max_partitions_contributed = max_contributions_per_partition = 2
        bound_result = self._run_l0_linf_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertFalse(bound_result)

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4)]
        max_partitions_contributed = max_contributions_per_partition = 2
        bound_result = self._run_l0_linf_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        expected_result = [(('pid1', 'pk2'), (2, 7, 25)),
                           (('pid1', 'pk1'), (2, 3, 5))]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_per_partition_bounding_applied(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid2", 'pk2', 6)]
        max_partitions_contributed, max_contributions_per_partition = 5, 2
        bound_result = self._run_l0_linf_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertEqual(3, len(bound_result))
        # Check contributions per partitions
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

        bound_result = self._run_l0_linf_contribution_bounding(
            input, max_partitions_contributed, max_contributions_per_partition)

        self.assertEqual(4, len(bound_result))
        # Check contributions per partitions
        self.assertTrue(
            all(
                map(
                    lambda op_val: op_val[1][0] <=
                    max_contributions_per_partition, bound_result)))
        # Check cross partition contributions
        dict_of_pid_to_pk = collections.defaultdict(lambda: [])
        for key, _ in bound_result:
            dict_of_pid_to_pk[key[0]].append(key[1])
        self.assertEqual(2, len(dict_of_pid_to_pk))
        self.assertTrue(
            all(
                map(
                    lambda key: len(dict_of_pid_to_pk[key]) <=
                    max_partitions_contributed, dict_of_pid_to_pk)))

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                 ("pid1", 'pk2', 4)]
        max_contributions = 4

        dp_engine = self._create_dp_engine_default()
        bound_result = list(
            dp_engine._bound_per_privacy_id_contributions(
                input,
                max_contributions=max_contributions,
                aggregator_fn=ContributionBoundersTest.aggregate_fn))

        expected_result = [(('pid1', 'pk2'), (2, 7, 25)),
                           (('pid1', 'pk1'), (2, 3, 5))]
        self.assertEqual(set(expected_result), set(bound_result))

    # def test_contribution_bounding_empty_col(self):
    #   input = []
    #   max_contributions = 4
    #
    #   dp_engine = self._create_dp_engine_default()
    #   bound_result = list(
    #       dp_engine._bound_per_privacy_id_contributions(
    #           input,
    #           max_contributions=max_contributions,
    #           aggregator_fn=ContributionBoundersTest.aggregate_fn))
    #
    #   self.assertFalse(bound_result)
