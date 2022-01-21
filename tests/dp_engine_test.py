import collections
from unittest.mock import patch
import numpy as np
import unittest
from absl.testing import absltest
from absl.testing import parameterized
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
from pipeline_dp.pipeline_backend import PipelineBackend
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


class DpEngineTest(parameterized.TestCase):
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

        self.assertEqual(3, len(bound_result))
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

    def test_aggregate_none(self):
        with self.assertRaises(Exception):
            pipeline_dp.DPEngine(None, None).aggregate(None, None, None)

    def test_check_aggregate_params(self):
        default_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )
        default_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT])

        test_cases = [
            {
                "desc": "None col",
                "col": None,
                "params": default_params,
                "data_extractor": default_extractors,
            },
            {
                "desc": "empty col",
                "col": [],
                "params": default_params,
                "data_extractor": default_extractors
            },
            {
                "desc": "none params",
                "col": [0],
                "params": None,
                "data_extractor": default_extractors,
            },
            {
                "desc":
                    "negative max_partitions_contributed",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=-1,
                        max_contributions_per_partition=1,
                        metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc":
                    "float max_partitions_contributed",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=1.5,
                        max_contributions_per_partition=1,
                        metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc":
                    "negative max_contributions_per_partition",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=1,
                        max_contributions_per_partition=-1,
                        metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc":
                    "float max_contributions_per_partition",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=1,
                        max_contributions_per_partition=1.5,
                        metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc":
                    "unspecified low",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=1,
                        max_contributions_per_partition=1,
                        metrics=[pipeline_dp.Metrics.SUM]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc":
                    "unspecified high",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=1,
                        max_contributions_per_partition=1,
                        metrics=[pipeline_dp.Metrics.SUM]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc":
                    "low > high",
                "col": [0],
                "params":
                    pipeline_dp.AggregateParams(
                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                        max_partitions_contributed=1,
                        max_contributions_per_partition=1,
                        min_value=1,
                        max_value=0,
                        metrics=[pipeline_dp.Metrics.SUM]),
                "data_extractor":
                    default_extractors,
            },
            {
                "desc": "None data_extractor",
                "col": [0],
                "params": default_params,
                "data_extractor": None,
            },
            {
                "desc": "data_extractor with an incorrect type",
                "col": [0],
                "params": default_params,
                "data_extractor": 1,
            },
        ]

        for test_case in test_cases:
            with self.assertRaises(Exception, msg=test_case["desc"]):
                budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-10)
                engine = pipeline_dp.DPEngine(
                    budget_accountant=budget_accountant,
                    ops=pipeline_dp.LocalPipelineOperations())
                engine.aggregate(test_case["col"], test_case["params"],
                                 test_case["data_extractor"])

    def test_aggregate_report(self):
        col = [[1], [2], [3], [3]]
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f"pid{x}",
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: x)
        params1 = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=3,
            max_contributions_per_partition=2,
            min_value=1,
            max_value=5,
            metrics=[
                pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT,
                pipeline_dp.Metrics.MEAN
            ],
        )
        params2 = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=1,
            max_contributions_per_partition=3,
            min_value=2,
            max_value=10,
            metrics=[
                pipeline_dp.Metrics.VAR, pipeline_dp.Metrics.SUM,
                pipeline_dp.Metrics.MEAN
            ],
            public_partitions=list(range(1, 40)),
        )

        select_partitions_params = SelectPrivatePartitionsParams(
            max_partitions_contributed=2)

        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-10)
        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalBackend())
        engine.aggregate(col, params1, data_extractor)
        engine.aggregate(col, params2, data_extractor)
        engine.select_private_partitions(col, select_partitions_params,
                                         data_extractor)
        self.assertEqual(3, len(engine._report_generators))  # pylint: disable=protected-access
        budget_accountant.compute_budgets()
        self.assertEqual(
            engine._report_generators[0].report(),
            "Differentially private: Computing <Metrics: ['privacy_id_count', 'count', 'mean']>"
            "\n1. Per-partition contribution bounding: randomly selected not more than 2 contributions"
            "\n2. Cross-partition contribution bounding: randomly selected not more than 3 partitions per user"
            "\n3. Private Partition selection: using Truncated Geometric method with (eps= 0.1111111111111111, delta = 1.1111111111111111e-11)"
        )
        self.assertEqual(
            engine._report_generators[1].report(),
            "Differentially private: Computing <Metrics: ['variance', 'sum', 'mean']>"
            "\n1. Public partition selection: dropped non public partitions"
            "\n2. Per-partition contribution bounding: randomly selected not more than 3 contributions"
            "\n3. Cross-partition contribution bounding: randomly selected not more than 1 partitions per user"
            "\n4. Adding empty partitions to public partitions that are missing in data"
        )
        self.assertEqual(
            engine._report_generators[2].report(),
            "Differentially private: Computing <Private Partitions>"
            "\n1. Private Partition selection: using Truncated Geometric method with (eps= 0.3333333333333333, delta = 3.3333333333333335e-11)"
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

        col = [[1], [2], [3], [3]]
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f"pid{x}",
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: x)

        mock_bound_contributions.return_value = [
            [("pid1", "pk1"), (1, [1])],
            [("pid2", "pk2"), (1, [1])],
            [("pid3", "pk3"), (1, [2])],
        ]

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalBackend())
        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)

        # Assert
        mock_bound_contributions.assert_called_with(
            unittest.mock.ANY, aggregator_params.max_partitions_contributed,
            aggregator_params.max_contributions_per_partition,
            unittest.mock.ANY)

    def _mock_and_assert_private_partitions(self, engine: pipeline_dp.DPEngine,
                                            col, min_users, expected_partitions,
                                            max_partitions_contributed):
        with patch(
                "pydp.algorithms.partition_selection.create_truncated_geometric_partition_strategy",
                new=_mock_partition_strategy_factory(
                    min_users)) as mock_factory:
            data_filtered = engine._select_private_partitions_internal(
                col, max_partitions_contributed)
            engine._budget_accountant.compute_budgets()
            self.assertListEqual(list(data_filtered), expected_partitions)

    @parameterized.named_parameters(
        dict(testcase_name='all_data_kept', min_users=1),
        dict(testcase_name='1 partition left', min_users=5),
        dict(testcase_name='empty result', min_users=20),
    )
    def test_select_private_partitions_internal(self, min_users):
        input_col = [("pk1", (3, None)), ("pk2", (10, None))]

        engine = self.create_dp_engine_default()
        expected_data_filtered = [x for x in input_col if x[1][0] > min_users]

        self._mock_and_assert_private_partitions(engine,
                                                 input_col,
                                                 min_users,
                                                 expected_data_filtered,
                                                 max_partitions_contributed=1)

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
                                      ops=pipeline_dp.LocalBackend())

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
                                      ops=pipeline_dp.LocalBackend())

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
                                      ops=pipeline_dp.LocalBackend())

        col = engine.select_private_partitions(col=col,
                                               params=params,
                                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        col = list(col)

        # Assert
        # Only one partition is retained, the one that has many unique _after_
        # applying the "max_partitions_contributed" bound is retained.
        self.assertEqual(["pk-many-contribs"], col)

    def test_check_select_private_partitions(self):
        """ Tests validation of parameters for select_private_partitions()"""
        default_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

        test_cases = [
            {
                "desc":
                    "None col",
                "col":
                    None,
                "params":
                    pipeline_dp.SelectPrivatePartitionsParams(
                        max_partitions_contributed=1,),
                "data_extractor":
                    default_extractor,
            },
            {
                "desc":
                    "empty col",
                "col": [],
                "params":
                    pipeline_dp.SelectPrivatePartitionsParams(
                        max_partitions_contributed=1,),
                "data_extractor":
                    default_extractor,
            },
            {
                "desc": "none params",
                "col": [0],
                "params": None,
                "data_extractor": default_extractor,
            },
            {
                "desc":
                    "negative max_partitions_contributed",
                "col": [0],
                "params":
                    pipeline_dp.SelectPrivatePartitionsParams(
                        max_partitions_contributed=-1,),
                "data_extractor":
                    default_extractor,
            },
            {
                "desc":
                    "float max_partitions_contributed",
                "col": [0],
                "params":
                    pipeline_dp.SelectPrivatePartitionsParams(
                        max_partitions_contributed=1.1,),
                "data_extractor":
                    default_extractor,
            },
            {
                "desc":
                    "None data_extractor",
                "col": [0],
                "params":
                    pipeline_dp.SelectPrivatePartitionsParams(
                        max_partitions_contributed=1,),
                "data_extractor":
                    None,
            },
            {
                "desc":
                    "Not a function data_extractor",
                "col": [0],
                "params":
                    pipeline_dp.SelectPrivatePartitionsParams(
                        max_partitions_contributed=1,),
                "data_extractor":
                    1,
            },
        ]

        for test_case in test_cases:
            with self.assertRaises(Exception, msg=test_case["desc"]):
                budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-10)
                engine = pipeline_dp.DPEngine(
                    budget_accountant=budget_accountant,
                    ops=pipeline_dp.LocalPipelineOperations())
                engine.select_private_partitions(test_case["col"],
                                                 test_case["params"],
                                                 test_case["data_extractor"])

    def test_aggregate_public_partitions_drop_non_public(self):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[
                agg.Metrics.COUNT, agg.Metrics.SUM, agg.Metrics.PRIVACY_ID_COUNT
            ],
            min_value=0,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            public_partitions=["pk0", "pk1", "pk10"])

        # Set an arbitrary budget, we are not interested in the DP outputs, only
        # the partition keys.
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=1e-10)

        # Input collection has 10 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        col = list(range(10))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: x)

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalBackend())

        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        col = list(col)
        partition_keys = [x[0] for x in col]
        # Assert

        # Only public partitions (0, 1, 2) should be kept and the rest of the
        # partitions should be dropped.
        self.assertEqual(["pk0", "pk1", "pk10"], partition_keys)

    def test_aggregate_public_partitions_add_empty_public_partitions(self):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[
                agg.Metrics.COUNT, agg.Metrics.SUM, agg.Metrics.PRIVACY_ID_COUNT
            ],
            min_value=0,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            public_partitions=["pk0", "pk10", "pk11"])

        # Set a high budget to add close to 0 noise.
        budget_accountant = NaiveBudgetAccountant(total_epsilon=100000,
                                                  total_delta=1 - 1e-10)

        # Input collection has 10 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        col = list(range(10))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: 1)

        engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                      ops=pipeline_dp.LocalBackend())

        col = engine.aggregate(col=col,
                               params=aggregator_params,
                               data_extractors=data_extractor)
        budget_accountant.compute_budgets()

        col = list(col)
        partition_keys = [x[0] for x in col]
        # Assert

        # Only public partitions ("pk0") should be kept and empty public
        # partitions ("pk10", "pk11") should be added.
        self.assertEqual(["pk0", "pk10", "pk11"], partition_keys)
        self.assertAlmostEqual(1, col[0][1][0])  # "pk0" COUNT ≈ 1
        self.assertAlmostEqual(1, col[0][1][1])  # "pk0" SUM ≈ 1
        self.assertAlmostEqual(1, col[0][1][2])  # "pk0" PRIVACY_ID_COUNT ≈ 1
        self.assertAlmostEqual(0, col[1][1][0])  # "pk10" COUNT ≈ 0
        self.assertAlmostEqual(0, col[1][1][1])  # "pk10" SUM ≈ 0
        self.assertAlmostEqual(0, col[1][1][2])  # "pk10" PRIVACY_ID_COUNT ≈ 0

    @staticmethod
    def create_dp_engine_default(accountant: NaiveBudgetAccountant = None,
                                 ops: PipelineBackend = None):
        if not accountant:
            accountant = NaiveBudgetAccountant(total_epsilon=1,
                                               total_delta=1e-10)
        if not ops:
            ops = pipeline_dp.LocalBackend()
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
            min_value=1,
            max_value=10,
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
            input, pipeline_dp.LocalBackend())

        self.assertEqual(5, len(list(output)))

    @unittest.skipIf(
        sys.platform == "win32" or
        (sys.version_info.minor <= 7 and sys.version_info.major == 3),
        "There are some problems with PySpark setup on older python and Windows"
    )
    def test_run_e2e_spark(self):
        import pyspark
        conf = pyspark.SparkConf()
        sc = pyspark.SparkContext.getOrCreate(conf=conf)
        input = sc.parallelize(list(range(10)))

        output = self.run_e2e_private_partition_selection_large_budget(
            input, pipeline_dp.SparkRDDBackend())

        self.assertEqual(5, len(output.collect()))

    def test_run_e2e_beam(self):
        with test_pipeline.TestPipeline() as p:
            input = p | "Create input" >> beam.Create(list(range(10)))

            output = self.run_e2e_private_partition_selection_large_budget(
                input, pipeline_dp.BeamBackend())

            beam_util.assert_that(output, beam_util.is_not_empty())


if __name__ == '__main__':
    absltest.main()
