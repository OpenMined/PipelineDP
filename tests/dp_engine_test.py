import collections
from unittest.mock import patch
import numpy as np
import unittest
import sys

import apache_beam as beam
import apache_beam.testing.test_pipeline as test_pipeline
import apache_beam.testing.util as beam_util

import pipeline_dp
from pipeline_dp.budget_accounting import NaiveBudgetAccountant
import pydp.algorithms.partition_selection as partition_selection
from pipeline_dp import aggregate_params as agg
from pipeline_dp.accumulator import CompoundAccumulatorFactory
from pipeline_dp.aggregate_params import SelectPrivatePartitionsParams
from pipeline_dp.accumulator import CountAccumulator
from pipeline_dp.report_generator import ReportGenerator
from pipeline_dp.pipeline_operations import PipelineOperations
"""DPEngine Test"""


class _MockPartitionStrategy(partition_selection.PartitionSelectionStrategy):

    def __init__(self, eps, delta, max_partitions_contributed, min_users):
        self.eps = eps
        self.delta = delta
        self.max_partitions_contributed = max_partitions_contributed
        self.min_users = min_users

    def should_keep(self, num_users: int) -> bool:
        return num_users > self.min_users


def _mock_partition_strategy_factory(min_users):

    def partition_strategy_factory(e, d, mpc):
        return _MockPartitionStrategy(e, d, mpc, min_users)

    return partition_strategy_factory


class _MockAccumulator(pipeline_dp.accumulator.Accumulator):

    def __init__(self, values_list: list = None) -> None:
        self.values_list = values_list or []

    @property
    def privacy_id_count(self):
        return len(self.values_list)

    def add_value(self, value):
        self.values_list.append(value)

    def add_accumulator(self,
                        accumulator: '_MockAccumulator') -> '_MockAccumulator':
        self.values_list.extend(accumulator.values_list)
        return self

    def compute_metrics(self):
        return self.values_list

    def __eq__(self, other: '_MockAccumulator') -> bool:
        return type(self) is type(other) and \
            sorted(self.values_list) == sorted(other.values_list)

    def __repr__(self) -> str:
        return f"MockAccumulator({self.values_list})"


class DpEngineTest(unittest.TestCase):
    aggregator_fn = lambda input_values: (len(input_values), np.sum(
        input_values), np.sum(np.square(input_values)))

    def test_contribution_bounding_empty_col(self):
        input_col = []
        max_partitions_contributed = 2
        max_contributions_per_partition = 2

        dp_engine = self.create_dp_engine_default()
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=DpEngineTest.aggregator_fn))

        self.assertFalse(bound_result)

    def test_contribution_bounding_bound_input_nothing_dropped(self):
        input_col = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                     ("pid1", 'pk2', 4)]
        max_partitions_contributed = 2
        max_contributions_per_partition = 2

        dp_engine = self.create_dp_engine_default()
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=DpEngineTest.aggregator_fn))

        expected_result = [(('pid1', 'pk2'), (2, 7, 25)),
                           (('pid1', 'pk1'), (2, 3, 5))]
        self.assertEqual(set(expected_result), set(bound_result))

    def test_contribution_bounding_per_partition_bounding_applied(self):
        input_col = [("pid1", 'pk1', 1), ("pid1", 'pk1', 2), ("pid1", 'pk2', 3),
                     ("pid1", 'pk2', 4), ("pid1", 'pk2', 5), ("pid2", 'pk2', 6)]
        max_partitions_contributed = 5
        max_contributions_per_partition = 2

        dp_engine = self.create_dp_engine_default()
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=DpEngineTest.aggregator_fn))

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

        dp_engine = self.create_dp_engine_default()
        bound_result = list(
            dp_engine._bound_contributions(
                input_col,
                max_partitions_contributed=max_partitions_contributed,
                max_contributions_per_partition=max_contributions_per_partition,
                aggregator_fn=DpEngineTest.aggregator_fn))

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

    @patch('pipeline_dp.accumulator._create_accumulator_factories')
    def test_aggregate_report(self, mock_create_accumulator_factories_function):
        col = [[1], [2], [3], [3]]
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f"pid{x}",
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: x)
        params1 = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
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
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
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

        select_partitions_params = SelectPrivatePartitionsParams(
            max_partitions_contributed=2)
        mock_create_accumulator_factories_function.return_value = [
            pipeline_dp.accumulator.CountAccumulatorFactory(None)
        ]
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-10)
        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalPipelineOperations())
        engine.aggregate(col, params1, data_extractor)
        engine.aggregate(col, params2, data_extractor)
        engine.select_private_partitions(col, select_partitions_params,
                                         data_extractor)
        self.assertEqual(len(engine._report_generators), 3)  # pylint: disable=protected-access
        budget_accountant.compute_budgets()
        self.assertEqual(
            engine._report_generators[0].report(),
            "Differentially private: Computing <Metrics: ['privacy_id_count', 'count', 'mean']>"
            "\n1. Per-partition contribution bounding: randomly selected not more than 2 contributions"
            "\n2. Cross-partition contribution bounding: randomly selected not more than 3 partitions per user"
            "\n3. Private Partition selection: using Truncated Geometric method with (eps= 0.5, delta = 5e-11)"
        )
        self.assertEqual(
            engine._report_generators[1].report(),
            "Differentially private: Computing <Metrics: ['variance', 'sum', 'mean']>"
            "\n1. Public partition selection: dropped non public partitions"
            "\n2. Per-partition contribution bounding: randomly selected not more than 3 contributions"
            "\n3. Cross-partition contribution bounding: randomly selected not more than 1 partitions per user"
        )
        self.assertEqual(
            engine._report_generators[2].report(),
            "Differentially private: Computing <Private Partitions>"
            "\n1. Private Partition selection: using Truncated Geometric method with (eps= 0.5, delta = 5e-11)"
        )

    @patch('pipeline_dp.DPEngine._bound_contributions')
    def test_aggregate_computation_graph_verification(self,
                                                      mock_bound_contributions):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[agg.Metrics.COUNT],
            max_partitions_contributed=5,
            max_contributions_per_partition=3)
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-10)
        accumulator_factory = CompoundAccumulatorFactory(
            params=aggregator_params, budget_accountant=budget_accountant)

        col = [[1], [2], [3], [3]]
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f"pid{x}",
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: x)

        mock_bound_contributions.return_value = [
            [("pid1", "pk1"),
             CountAccumulator(params=None, values=[1])],
            [("pid2", "pk2"),
             CountAccumulator(params=None, values=[1])],
            [("pid3", "pk3"),
             CountAccumulator(params=None, values=[2])],
        ]

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalPipelineOperations())
        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)

        # Assert
        mock_bound_contributions.assert_called_with(
            unittest.mock.ANY, aggregator_params.max_partitions_contributed,
            aggregator_params.max_contributions_per_partition,
            unittest.mock.ANY)

    def _mock_and_assert_private_partitions(self, engine: pipeline_dp.DPEngine,
                                            groups, min_users,
                                            expected_partitions,
                                            max_partitions_contributed):
        with patch(
                "pydp.algorithms.partition_selection.create_truncated_geometric_partition_strategy",
                new=_mock_partition_strategy_factory(
                    min_users)) as mock_factory:
            data_filtered = engine._select_private_partitions_internal(
                groups, max_partitions_contributed)
            engine._budget_accountant.compute_budgets()
            self.assertListEqual(list(data_filtered), expected_partitions)

    def test_select_private_partitions_internal(self):
        input_col = [("pid1", ('pk1', 1)), ("pid1", ('pk1', 2)),
                     ("pid1", ('pk2', 3)), ("pid1", ('pk2', 4)),
                     ("pid1", ('pk2', 5)), ("pid1", ('pk3', 6)),
                     ("pid1", ('pk4', 7)), ("pid2", ('pk4', 8))]
        max_partitions_contributed = 3
        engine = self.create_dp_engine_default()
        groups = engine._ops.group_by_key(input_col, None)
        groups = engine._ops.map_values(groups,
                                        lambda group: _MockAccumulator(group))
        groups = list(groups)
        expected_data_filtered = [("pid1",
                                   _MockAccumulator([
                                       ('pk1', 1),
                                       ('pk1', 2),
                                       ('pk2', 3),
                                       ('pk2', 4),
                                       ('pk2', 5),
                                       ('pk3', 6),
                                       ('pk4', 7),
                                   ])),
                                  ("pid2", _MockAccumulator([('pk4', 8)]))]
        self._mock_and_assert_private_partitions(engine, groups, 0,
                                                 expected_data_filtered,
                                                 max_partitions_contributed)
        expected_data_filtered = [
            ("pid1",
             _MockAccumulator([
                 ('pk1', 1),
                 ('pk1', 2),
                 ('pk2', 3),
                 ('pk2', 4),
                 ('pk2', 5),
                 ('pk3', 6),
                 ('pk4', 7),
             ])),
        ]
        self._mock_and_assert_private_partitions(engine, groups, 3,
                                                 expected_data_filtered,
                                                 max_partitions_contributed)
        expected_data_filtered = []
        self._mock_and_assert_private_partitions(engine, groups, 100,
                                                 expected_data_filtered,
                                                 max_partitions_contributed)

    def test_aggregate_private_partition_selection_keep_everything(self):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[agg.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)
        # Set a large budget for having the small noise and keeping all
        # partition keys.
        budget_accountant = NaiveBudgetAccountant(total_epsilon=100000,
                                                  total_delta=1e-10)

        col = list(range(10)) + list(range(100, 120))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x//100}",
            value_extractor=lambda x: None)

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalPipelineOperations())

        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        col = list(col)

        # Assert
        approximate_expected = {"pk0": 10, "pk1": 20}
        self.assertEqual(2, len(col))  # all partition keys are kept.
        for pk, anonymized_count in col:
            self.assertAlmostEqual(approximate_expected[pk],
                                   anonymized_count[0],
                                   delta=1e-3)

    def test_aggregate_private_partition_selection_drop_many(self):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[agg.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        # Set a small budget for dropping most partition keys.
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-10)

        # Input collection has 100 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        col = list(range(100))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: None)

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalPipelineOperations())

        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        col = list(col)

        # Assert

        # Most partition should be dropped by private partition selection.
        # This tests is non-deterministic, but it should pass with probability
        # very close to 1.
        self.assertLess(len(col), 5)

    def test_select_private_partitions(self):
        # This test is probabilistic, but the parameters were chosen to ensure
        # the test has passed at least 10000 runs.

        # Arrange
        params = SelectPrivatePartitionsParams(max_partitions_contributed=1)

        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-5)

        # Generate dataset as a list of (user, partition_key) tuples.
        # There partitions are generated to reflect several scenarios.

        # A partition with sufficient amount of users.
        col = [(u, "pk-many-contribs") for u in range(25)]

        # A partition with many contributions, but only a few unique users.
        col += [(100 + u // 10, "pk-many-contribs-few-users") for u in range(30)
               ]

        # A partition with few contributions.
        col += [(200 + u, "pk-few-contribs") for u in range(3)]

        # Generating 30 partitions, each with the same group of 25 users
        # 25 users is sufficient to keep the partition, but because of
        # contribution bounding, much less users per partition will be kept.
        for i in range(30):
            col += [(500 + u, f"few-contribs-after-bound{i}") for u in range(25)
                   ]

        col = list(col)
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1])

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalPipelineOperations())

        col = engine.select_private_partitions(col=col,
                                               params=params,
                                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        col = list(col)

        # Assert
        # Only one partition is retained, the one that has many unique _after_
        # applying the "max_partitions_contributed" bound is retained.
        self.assertEqual(["pk-many-contribs"], col)

    @staticmethod
    def create_dp_engine_default(accountant: NaiveBudgetAccountant = None,
                                 ops: PipelineOperations = None):
        if not accountant:
            accountant = NaiveBudgetAccountant(total_epsilon=1,
                                               total_delta=1e-10)
        if not ops:
            ops = pipeline_dp.LocalPipelineOperations()
        dp_engine = pipeline_dp.DPEngine(accountant, ops)
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            metrics=[],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)
        dp_engine._report_generators.append(ReportGenerator(aggregator_params))
        dp_engine._add_report_stage("DP Engine Test")
        return dp_engine

    @staticmethod
    def run_e2e_private_partition_selection_large_budget(col, ops):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            metrics=[agg.Metrics.COUNT, agg.Metrics.SUM],
            low=1,
            high=10,
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        # Set a large budget for having the small noise and keeping all
        # partition keys.
        budget_accountant = NaiveBudgetAccountant(total_epsilon=100000,
                                                  total_delta=1)

        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x//2}",
            value_extractor=lambda x: x)

        engine = pipeline_dp.DPEngine(budget_accountant, ops)

        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        return col

    def test_run_e2e_local(self):
        input = list(range(10))

        output = self.run_e2e_private_partition_selection_large_budget(
            input, pipeline_dp.LocalPipelineOperations())

        self.assertEqual(5, len(list(output)))

    @unittest.skipIf(sys.platform == "win32",
                     "There are some problems with PySpark setup on Windows")
    def test_run_e2e_spark(self):
        import pyspark
        conf = pyspark.SparkConf()
        sc = pyspark.SparkContext.getOrCreate(conf=conf)
        input = sc.parallelize(list(range(10)))

        output = self.run_e2e_private_partition_selection_large_budget(
            input, pipeline_dp.SparkRDDOperations())

        self.assertEqual(5, len(output.collect()))

    def test_run_e2e_beam(self):
        with test_pipeline.TestPipeline() as p:
            input = p | "Create input" >> beam.Create(list(range(10)))

            output = self.run_e2e_private_partition_selection_large_budget(
                input, pipeline_dp.BeamOperations())

            beam_util.assert_that(output, beam_util.is_not_empty())


if __name__ == '__main__':
    unittest.main()
