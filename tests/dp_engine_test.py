import collections
import numpy as np
import unittest

import pipeline_dp
"""DPEngine Test"""


class dp_engineTest(unittest.TestCase):
    aggregator_fn = lambda input_values: (len(input_values), np.sum(
        input_values), np.sum(np.square(input_values)))

    def test_contribution_bounding_empty_col(self):
        input_col = []
        max_partitions_contributed = 2
        max_contributions_per_partition = 2

        dp_engine = pipeline_dp.DPEngine(
            pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
            pipeline_dp.LocalPipelineOperations())
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=dp_engineTest.aggregator_fn))

        self.assertFalse(bound_result)

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input_col = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                     ("pid1", 'pk2', 4)]
        max_partitions_contributed = 2
        max_contributions_per_partition = 2

        dp_engine = pipeline_dp.DPEngine(
            pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
            pipeline_dp.LocalPipelineOperations())
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=dp_engineTest.aggregator_fn))

        expected_result = [(('pid1', 'pk2'), (2, 7, 25)),
                           (('pid1', 'pk1'), (2, 3, 5))]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_per_partition_bounding_applied(self):
        input_col = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                     ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid2", 'pk2', 6)]
        max_partitions_contributed = 5
        max_contributions_per_partition = 2

        dp_engine = pipeline_dp.DPEngine(
            pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
            pipeline_dp.LocalPipelineOperations())
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=dp_engineTest.aggregator_fn))

        self.assertEqual(len(bound_result), 3)
        # Check contributions per partitions
        self.assertTrue(
            all(
                map(
                    lambda op_val: op_val[1][0] <=
                    max_contributions_per_partition, bound_result)))

    def test_contribution_bounding_cross_partition_bounding_applied(self):
        input_col = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                     ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid1", 'pk3', 6),
                     ("pid1", 'pk4', 7), ("pid2", 'pk4', 8)]
        max_partitions_contributed = 3
        max_contributions_per_partition = 5

        dp_engine = pipeline_dp.DPEngine(
            pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
            pipeline_dp.LocalPipelineOperations())
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=dp_engineTest.aggregator_fn))

        self.assertEqual(len(bound_result), 4)
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
        self.assertEqual(len(dict_of_pid_to_pk), 2)
        self.assertTrue(
            all(
                map(
                    lambda key: len(dict_of_pid_to_pk[key]) <=
                    max_partitions_contributed, dict_of_pid_to_pk)))

    def test_aggregate_none(self):
        self.assertIsNone(
            pipeline_dp.DPEngine(None, None).aggregate(None, None, None))

    def test_aggregate_report(self):
        params1 = pipeline_dp.AggregateParams(
            max_partitions_contributed=3,
            max_contributions_per_partition=2,
            low=1,
            high=5,
            metrics=[
                pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT,
                pipeline_dp.Metrics.MEAN
            ],
        )
        params2 = pipeline_dp.AggregateParams(
            max_partitions_contributed=1,
            max_contributions_per_partition=3,
            low=2,
            high=10,
            metrics=[
                pipeline_dp.Metrics.VAR, pipeline_dp.Metrics.SUM,
                pipeline_dp.Metrics.MEAN
            ],
            public_partitions=list(range(1, 40)),
        )
        engine = pipeline_dp.DPEngine(None, None)
        engine.aggregate(None, params1, None)
        engine.aggregate(None, params2, None)
        self.assertEqual(len(engine._report_generators), 2)  # pylint: disable=protected-access


if __name__ == '__main__':
    unittest.main()
