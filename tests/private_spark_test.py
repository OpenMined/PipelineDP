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
import pyspark
from pyspark import SparkContext
from absl.testing import parameterized
from unittest.mock import patch
import unittest
import sys
import collections
import pipeline_dp
from pipeline_dp import aggregate_params as ap
from pipeline_dp import budget_accounting
from pipeline_dp import private_spark


@unittest.skipIf(sys.version_info.minor <= 7 and sys.version_info.major == 3,
                 "There are some problems with PySpark setup on older python.")
class PrivateRDDTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf()
        cls.sc = SparkContext.getOrCreate(conf=conf)

    @classmethod
    def value_per_key_within_tolerance(self, expected, actual, tolerance):
        return abs(actual - expected) <= tolerance

    @classmethod
    def to_dict(self, tuples):
        di = {}
        for a, b in tuples:
            di.setdefault(a, b)
        return di

    def test_map(self):
        data = [(1, 11), (2, 12)]
        dist_data = PrivateRDDTest.sc.parallelize(data)
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.PrivateRDD(dist_data, budget_accountant,
                                        privacy_id_extractor)

        result = prdd.map(lambda x: (x[0], x[1] * 2))

        self.assertEqual(result._rdd.collect(), [(1, (1, 22)), (2, (2, 24))])
        self.assertEqual(result._budget_accountant, prdd._budget_accountant)

    def test_flatmap(self):
        data = [(1, 11), (2, 12)]
        dist_data = PrivateRDDTest.sc.parallelize(data)
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.PrivateRDD(dist_data, budget_accountant,
                                        privacy_id_extractor)

        result = prdd.flat_map(lambda x: [(x[0], x[1] * 2),
                                          (x[0], x[1] * 2 + 1)])

        self.assertEqual(result._rdd.collect(), [(1, (1, 22)), (1, (1, 23)),
                                                 (2, (2, 24)), (2, (2, 25))])
        self.assertEqual(result._budget_accountant, prdd._budget_accountant)

    @parameterized.parameters(True, False)
    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_variance_calls_aggregate_with_correct_params(
            self, contribution_bounds_already_enforced: bool, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, 0.0, "pk1"),
                                                   (2, 10.0, "pk1")])
        MetricsTuple = collections.namedtuple('MetricsTuple', ['variance'])
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([
            ("pk1", MetricsTuple(variance=25.0))
        ])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[1]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        variance_params = ap.VarianceParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            min_value=1.5,
            max_value=5.78,
            budget_weight=1.1,
            partition_extractor=lambda x: x[0],
            value_extractor=lambda x: x,
            contribution_bounds_already_enforced=
            contribution_bounds_already_enforced)

        # Act
        actual_result = prdd.variance(variance_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]

        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = ap.AggregateParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            metrics=[ap.Metrics.VARIANCE],
            max_partitions_contributed=variance_params.
            max_partitions_contributed,
            max_contributions_per_partition=variance_params.
            max_contributions_per_partition,
            min_value=variance_params.min_value,
            max_value=variance_params.max_value,
            budget_weight=variance_params.budget_weight,
            contribution_bounds_already_enforced=
            contribution_bounds_already_enforced)
        self.assertEqual(args[1], params)

        self.assertEqual(actual_result.collect(), [("pk1", 25.0)])

    def test_variance_returns_sensible_result(self):
        # Arrange
        col = [(u, "pk1", -100) for u in range(30)]
        col += [(u + 30, "pk1", 100) for u in range(10)]

        dist_data = PrivateRDDTest.sc.parallelize(col)
        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        variance_params = ap.VarianceParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            min_value=1.55,  # -100 should be clipped to this value
            max_value=2.7889,  # 100 should be clipped to this value
            budget_weight=1,
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.variance(variance_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pk1": 0.288}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, variance in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(variance,
                                                    expected_result_dict[pk],
                                                    0.1))

    def test_variance_with_public_partitions_returns_sensible_result(self):
        # Arrange
        col = [(u, "pubK1", -100) for u in range(30)]
        col += [(u + 30, "pubK1", 100) for u in range(10)]
        col += [(u + 40, "privK1", 100) for u in range(30)]

        dist_data = PrivateRDDTest.sc.parallelize(col)
        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=8000, total_delta=0.9999999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        variance_params = ap.VarianceParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            min_value=1.55,  # -100 should be clipped to this value
            max_value=2.7889,  # 100 should be clipped to this value
            budget_weight=1,
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.variance(variance_params,
                                      public_partitions=["pubK1", "pubK2"])
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pubK1": 0.288, "pubK2": 0.0}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, variance in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(variance,
                                                    expected_result_dict[pk],
                                                    0.1))

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_mean_calls_aggregate_with_correct_params(self, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, 2.0, "pk1"),
                                                   (2, 2.0, "pk1")])
        MetricsTuple = collections.namedtuple('MetricsTuple', ['mean'])
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([
            ("pk1", MetricsTuple(mean=2.0))
        ])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[1]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        mean_params = ap.MeanParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                    max_partitions_contributed=2,
                                    max_contributions_per_partition=3,
                                    min_value=1.5,
                                    max_value=5.78,
                                    budget_weight=1.1,
                                    partition_extractor=lambda x: x[0],
                                    value_extractor=lambda x: x)

        # Act
        actual_result = prdd.mean(mean_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]

        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = ap.AggregateParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            metrics=[ap.Metrics.MEAN],
            max_partitions_contributed=mean_params.max_partitions_contributed,
            max_contributions_per_partition=mean_params.
            max_contributions_per_partition,
            min_value=mean_params.min_value,
            max_value=mean_params.max_value,
            budget_weight=mean_params.budget_weight)
        self.assertEqual(args[1], params)

        self.assertEqual(actual_result.collect(), [("pk1", 2.0)])

    def test_mean_returns_sensible_result(self):
        # Arrange
        col = [(u, "pk1", -100) for u in range(30)]
        col += [(u + 30, "pk1", 100) for u in range(10)]

        dist_data = PrivateRDDTest.sc.parallelize(col)
        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        mean_params = ap.MeanParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                    max_partitions_contributed=2,
                                    max_contributions_per_partition=3,
                                    min_value=1.55,
                                    max_value=2.7889,
                                    budget_weight=1,
                                    partition_extractor=lambda x: x[1],
                                    value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.mean(mean_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pk1": 1.859}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, mean in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(mean,
                                                    expected_result_dict[pk],
                                                    0.1))

    def test_mean_with_public_partitions_returns_sensible_result(self):
        # Arrange
        col = [(u, "pubK1", -100) for u in range(30)]
        col += [(u + 30, "pubK1", 100) for u in range(10)]
        col += [(u + 40, "privK1", 100) for u in range(30)]

        dist_data = PrivateRDDTest.sc.parallelize(col)
        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=8000, total_delta=0.999999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        mean_params = ap.MeanParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                    max_partitions_contributed=2,
                                    max_contributions_per_partition=3,
                                    min_value=1.55,
                                    max_value=2.789,
                                    budget_weight=1,
                                    partition_extractor=lambda x: x[1],
                                    value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.mean(mean_params,
                                  public_partitions=["pubK1", "pubK2"])
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pubK1": 1.859, "pubK2": 2.1695}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, mean in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(mean,
                                                    expected_result_dict[pk],
                                                    0.1))

    @parameterized.parameters(True, False)
    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_sum_calls_aggregate_with_correct_params(
            self, contribution_bounds_already_enforced: bool, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, 1.0, "pk1"),
                                                   (2, 2.0, "pk1")])
        MetricsTuple = collections.namedtuple('MetricsTuple', ['sum'])
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([
            ("pk1", MetricsTuple(sum=3.0))
        ])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[1]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        sum_params = ap.SumParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                  max_partitions_contributed=2,
                                  max_contributions_per_partition=3,
                                  min_value=1.55,
                                  max_value=2.7889,
                                  budget_weight=1.1,
                                  partition_extractor=lambda x: x[0],
                                  value_extractor=lambda x: x,
                                  contribution_bounds_already_enforced=
                                  contribution_bounds_already_enforced)

        # Act
        actual_result = prdd.sum(sum_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]

        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = ap.AggregateParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            metrics=[ap.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.
            max_contributions_per_partition,
            min_value=sum_params.min_value,
            max_value=sum_params.max_value,
            budget_weight=sum_params.budget_weight,
            contribution_bounds_already_enforced=
            contribution_bounds_already_enforced)
        self.assertEqual(args[1], params)

        self.assertEqual(actual_result.collect(), [("pk1", 3.0)])

    def test_sum_returns_sensible_result(self):
        # Arrange
        col = [(f"{u}", "pk1", 100.0) for u in range(30)]
        col += [(f"{u + 30}", "pk1", -100.0) for u in range(30)]

        dist_data = PrivateRDDTest.sc.parallelize(col)
        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        sum_params = ap.SumParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                  max_partitions_contributed=2,
                                  max_contributions_per_partition=3,
                                  min_value=1.55,
                                  max_value=2.7889,
                                  budget_weight=1,
                                  partition_extractor=lambda x: x[1],
                                  value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.sum(sum_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pk1": 130.167}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, sum in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(sum,
                                                    expected_result_dict[pk],
                                                    5.0))

    def test_sum_calls_with_public_partitions_returns_sensible_result(self):
        # Arrange
        col = [(f"{u}", "pubK1", 100.0) for u in range(30)]
        col += [(f"{u + 30}", "pubK1", -100.0) for u in range(30)]
        col += [(f"{u + 60}", "privK1", 100.0) for u in range(30)]
        dist_data = PrivateRDDTest.sc.parallelize(col)
        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        sum_params = ap.SumParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                  max_partitions_contributed=2,
                                  max_contributions_per_partition=3,
                                  min_value=1.55,
                                  max_value=2.7889,
                                  budget_weight=1,
                                  partition_extractor=lambda x: x[1],
                                  value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.sum(sum_params,
                                 public_partitions=["pubK1", "pubK2"])
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pubK2": 0.0, "pubK1": 130.167}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, sum in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(sum,
                                                    expected_result_dict[pk],
                                                    5.0))

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_count_calls_aggregate_with_correct_params(self, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, "pk1"), (2, "pk1")])
        MetricsTuple = collections.namedtuple('MetricsTuple', ['count'])
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([
            ("pk1", MetricsTuple(count=2))
        ])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)

        count_params = ap.CountParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                      max_partitions_contributed=2,
                                      max_contributions_per_partition=3,
                                      budget_weight=1,
                                      partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.count(count_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]
        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = ap.AggregateParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            metrics=[ap.Metrics.COUNT],
            max_partitions_contributed=count_params.max_partitions_contributed,
            max_contributions_per_partition=count_params.
            max_contributions_per_partition)
        self.assertEqual(args[1], params)

        self.assertEqual(actual_result.collect(), [("pk1", 2)])

    def test_count_returns_sensible_result(self):
        # Arrange
        col = [(u, "pk1") for u in range(30)]
        dist_data = PrivateRDDTest.sc.parallelize(col)

        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)

        count_params = ap.CountParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                      max_partitions_contributed=2,
                                      max_contributions_per_partition=3,
                                      budget_weight=1,
                                      partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.count(count_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pk1": 30.0}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, count in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(count,
                                                    expected_result_dict[pk],
                                                    5.0))

    def test_count_calls_with_public_partitions_returns_sensible_result(self):
        # Arrange
        col = [(u, "pubK1") for u in range(30)]
        col += [(u, "privK1") for u in range(30)]
        dist_data = PrivateRDDTest.sc.parallelize(col)

        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)

        count_params = ap.CountParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                      max_partitions_contributed=2,
                                      max_contributions_per_partition=3,
                                      budget_weight=1,
                                      partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.count(count_params,
                                   public_partitions=["pubK1", "pubK2"])
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pubK2": 0.0, "pubK1": 30.0}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, count in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(count,
                                                    expected_result_dict[pk],
                                                    5.0))

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_privacy_id_count_calls_aggregate_with_correct_params(
            self, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, "pk1"), (2, "pk1")])
        MetricsTuple = collections.namedtuple('MetricsTuple',
                                              ['privacy_id_count'])
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([
            ("pk1", MetricsTuple(privacy_id_count=2))
        ])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        privacy_id_count_params = ap.PrivacyIdCountParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            budget_weight=1,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.privacy_id_count(privacy_id_count_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]

        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = ap.AggregateParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            metrics=[ap.Metrics.PRIVACY_ID_COUNT],
            max_partitions_contributed=privacy_id_count_params.
            max_partitions_contributed,
            max_contributions_per_partition=1)
        self.assertEqual(args[1], params)

        self.assertEqual([("pk1", 2)], actual_result.collect())

    def test_privacy_id_count_returns_sensible_result(self):
        # Arrange
        col = [(u, "pk1") for u in range(30)]
        dist_data = PrivateRDDTest.sc.parallelize(col)
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        privacy_id_count_params = ap.PrivacyIdCountParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            budget_weight=1,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.privacy_id_count(privacy_id_count_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pk1": 30.0}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, count in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(count,
                                                    expected_result_dict[pk],
                                                    5.0))

    def test_privacy_id_count_with_public_partitions_returns_sensible_result(
            self):
        # Arrange
        col = [(u, "pubK1") for u in range(30)]
        col += [(u, "privK1") for u in range(30)]
        dist_data = PrivateRDDTest.sc.parallelize(col)

        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        privacy_id_count_params = ap.PrivacyIdCountParams(
            noise_kind=ap.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            budget_weight=1,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.privacy_id_count(
            privacy_id_count_params, public_partitions=["pubK1", "pubK2"])
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pubK2": 0.0, "pubK1": 30.0}
        actual_result_dict = self.to_dict(actual_result.collect())

        for pk, sum in actual_result_dict.items():
            self.assertTrue(
                self.value_per_key_within_tolerance(sum,
                                                    expected_result_dict[pk],
                                                    5.0))

    @patch('pipeline_dp.dp_engine.DPEngine.select_partitions')
    def test_select_partitions_calls_select_partitions_with_correct_params(
            self, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, "pk1"), (2, "pk2")])
        expected_result_partitions = ["pk1", "pk2"]
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize(
            expected_result_partitions)
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=1, total_delta=0.01)
        max_partitions_contributed = 2

        def privacy_id_extractor(x):
            return x[0]

        def partition_extractor(x):
            return {x[1]}

        # Act
        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)

        select_partitions_params = ap.SelectPartitionsParams(
            max_partitions_contributed=max_partitions_contributed)
        actual_result = prdd.select_partitions(select_partitions_params,
                                               partition_extractor)

        # Assert
        mock_aggregate.assert_called_once()
        actual_args = mock_aggregate.call_args[0]
        actual_rdd = actual_args[0].collect()
        actual_select_partition_params = actual_args[1]

        self.assertListEqual(actual_rdd, [(1, (1, "pk1")), (2, (2, "pk2"))])

        self.assertEqual(
            actual_select_partition_params.max_partitions_contributed,
            max_partitions_contributed)
        self.assertEqual(actual_result.collect(), expected_result_partitions)

    def test_select_partitions_returns_sensible_result(self):
        # Arrange
        col = [(u, "pk1") for u in range(50)]
        col += [(50 + u, "pk2") for u in range(50)]
        dist_data = PrivateRDDTest.sc.parallelize(col)

        # Use very high epsilon and delta to minimize noise and test
        # flakiness.
        budget_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=800, total_delta=0.999)
        max_partitions_contributed = 2

        def privacy_id_extractor(x):
            return x[0]

        def partition_extractor(x):
            return x[1]

        # Act
        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)

        select_partitions_params = ap.SelectPartitionsParams(
            max_partitions_contributed=max_partitions_contributed)
        actual_result = prdd.select_partitions(select_partitions_params,
                                               partition_extractor)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        self.assertEqual(sorted(actual_result.collect()), ["pk1", "pk2"])

    def test_private_rdd_methods_cache_dp_output_across_actions(self):
        col = [(u, "p", float(u)) for u in range(100)]
        dist_data = PrivateRDDTest.sc.parallelize(col)

        accountant = budget_accounting.NaiveBudgetAccountant(total_epsilon=6.0,
                                                             total_delta=0.1)
        prdd = private_spark.make_private(dist_data, accountant, lambda r: r[0])
        common_args = dict(noise_kind=ap.NoiseKind.LAPLACE,
                           max_partitions_contributed=1,
                           partition_extractor=lambda r: r[1])
        value_args = dict(**common_args,
                          max_contributions_per_partition=1,
                          min_value=0.0,
                          max_value=10.0,
                          value_extractor=lambda r: r[2])
        results = [
            prdd.count(ap.CountParams(**common_args,
                                      max_contributions_per_partition=1),
                       public_partitions=["p"]),
            prdd.sum(ap.SumParams(**value_args), public_partitions=["p"]),
            prdd.mean(ap.MeanParams(**value_args), public_partitions=["p"]),
            prdd.variance(ap.VarianceParams(**value_args),
                          public_partitions=["p"]),
            prdd.privacy_id_count(ap.PrivacyIdCountParams(**common_args),
                                  public_partitions=["p"]),
        ]
        # Narrow transform chained BEFORE compute_budgets() must not bypass cache.
        pre_filtered_count = results[0].filter(lambda kv: kv[0] == "p")
        accountant.compute_budgets()

        for res in results + [pre_filtered_count]:
            first = res.collect()
            for _ in range(5):
                self.assertEqual(res.collect(), first)
                self.assertEqual(res.count(), len(first))
        self.assertEqual(pre_filtered_count.collect(), results[0].collect())
        self.assertEqual(results[0].filter(lambda kv: kv[0] == "p").collect(),
                         results[0].collect())

        sp_col = [(u, "safe") for u in range(60)] + [
            (60 + b * 4 + j, f"border_{b}") for b in range(15) for j in range(4)
        ]
        sp_accountant = budget_accounting.NaiveBudgetAccountant(
            total_epsilon=2.0, total_delta=0.2)
        sp_prdd = private_spark.make_private(
            PrivateRDDTest.sc.parallelize(sp_col), sp_accountant,
            lambda r: r[0])
        sp_result = sp_prdd.select_partitions(
            ap.SelectPartitionsParams(max_partitions_contributed=1),
            lambda r: r[1])
        self.assertFalse(sp_result.is_cached)
        sp_accountant.compute_budgets()
        self.assertTrue(sp_result.is_cached)
        first_sp = sorted(sp_result.collect())
        self.assertIn("safe", first_sp)
        for _ in range(5):
            self.assertEqual(sorted(sp_result.collect()), first_sp)

    def test_dp_engine_spark_caches_aggregate_select_partitions_and_noise(self):
        backend = pipeline_dp.SparkRDDBackend(PrivateRDDTest.sc)
        accountant = budget_accounting.NaiveBudgetAccountant(total_epsilon=4.0,
                                                             total_delta=0.4)
        dp_engine = pipeline_dp.DPEngine(accountant, backend)

        agg_rdd = PrivateRDDTest.sc.parallelize([
            (u, "p", float(u % 5 + 1)) for u in range(50)
        ])
        extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda r: r[0],
            partition_extractor=lambda r: r[1],
            value_extractor=lambda r: r[2])
        agg_params = ap.AggregateParams(
            noise_kind=ap.NoiseKind.LAPLACE,
            metrics=[ap.Metrics.COUNT, ap.Metrics.SUM],
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            min_value=0.0,
            max_value=5.0)
        agg_result = dp_engine.aggregate(agg_rdd,
                                         agg_params,
                                         extractors,
                                         public_partitions=["p"])
        pre_counts = agg_result.mapValues(lambda v: v.count)
        pre_sums = agg_result.mapValues(lambda v: v.sum)

        sp_rows = [(u, "safe") for u in range(80)] + [
            (80 + b * 4 + j, f"border_{b}") for b in range(20) for j in range(4)
        ]
        sp_rdd = PrivateRDDTest.sc.parallelize(sp_rows)
        sp_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda r: r[0],
            partition_extractor=lambda r: r[1])
        sp_result = dp_engine.select_partitions(
            sp_rdd, ap.SelectPartitionsParams(max_partitions_contributed=1),
            sp_extractors)

        wg_result = dp_engine.select_partitions(
            sp_rdd,
            ap.SelectPartitionsParams(
                max_partitions_contributed=1,
                partition_selection_strategy=ap.PartitionSelectionStrategy.
                WEIGHTED_GAUSSIAN_THRESHOLDING), sp_extractors)

        noise_rdd = PrivateRDDTest.sc.parallelize([("p", 10.0)])
        noise_result = dp_engine.add_dp_noise(
            noise_rdd,
            ap.AddDPNoiseParams(noise_kind=ap.NoiseKind.LAPLACE,
                                l1_sensitivity=1.0))

        for res in (agg_result, sp_result, wg_result, noise_result):
            self.assertFalse(res.is_cached)

        accountant.compute_budgets()

        for res in (agg_result, sp_result, wg_result, noise_result):
            self.assertTrue(res.is_cached)
            first = sorted(res.collect())
            if res in (sp_result, wg_result):
                self.assertIn("safe", first)
            for _ in range(5):
                self.assertEqual(sorted(res.collect()), first)

        first_counts = pre_counts.collect()
        first_sums = pre_sums.collect()
        self.assertEqual(first_counts[0][1], agg_result.collect()[0][1].count)
        self.assertEqual(first_sums[0][1], agg_result.collect()[0][1].sum)
        for _ in range(5):
            self.assertEqual(pre_counts.collect(), first_counts)
            self.assertEqual(pre_sums.collect(), first_sums)

    def test_dp_engine_spark_with_pld_accountant_caches_across_actions(self):
        backend = pipeline_dp.SparkRDDBackend(PrivateRDDTest.sc)
        accountant = budget_accounting.PLDBudgetAccountant(total_epsilon=2.0,
                                                           total_delta=1e-3)
        dp_engine = pipeline_dp.DPEngine(accountant, backend)
        agg_rdd = PrivateRDDTest.sc.parallelize([
            (u, "p", 1.0) for u in range(50)
        ])
        extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda r: r[0],
            partition_extractor=lambda r: r[1],
            value_extractor=lambda r: r[2])
        agg_result = dp_engine.aggregate(agg_rdd,
                                         ap.AggregateParams(
                                             noise_kind=ap.NoiseKind.LAPLACE,
                                             metrics=[ap.Metrics.COUNT],
                                             max_partitions_contributed=1,
                                             max_contributions_per_partition=1),
                                         extractors,
                                         public_partitions=["p"])
        self.assertFalse(agg_result.is_cached)
        accountant.compute_budgets()
        self.assertTrue(agg_result.is_cached)
        first = agg_result.collect()
        for _ in range(5):
            self.assertEqual(agg_result.collect(), first)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


if __name__ == '__main__':
    unittest.main()
