import pyspark
from pyspark import SparkContext
from unittest.mock import patch
import unittest
import sys

import pipeline_dp
from pipeline_dp import aggregate_params as agg
from pipeline_dp import budget_accounting, private_spark


@unittest.skipIf(
    sys.platform == "win32" or
    (sys.version_info.minor <= 7 and sys.version_info.major == 3),
    "There are some problems with PySpark setup on older python and Windows")
class PrivateRDDTest(unittest.TestCase):

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

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_sum_calls_aggregate_with_correct_params(self, mock_aggregate):
        # Arrange
        dist_data = PrivateRDDTest.sc.parallelize([(1, 1.0, "pk1"),
                                                   (2, 2.0, "pk1")])
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([(3.0,
                                                                      ["pk1"])])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[1]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        sum_params = agg.SumParams(noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                   max_partitions_contributed=2,
                                   max_contributions_per_partition=3,
                                   min_value=1,
                                   max_value=5,
                                   budget_weight=1,
                                   public_partitions=None,
                                   partition_extractor=lambda x: x[0],
                                   value_extractor=lambda x: x)

        # Act
        actual_result = prdd.sum(PrivateRDDTest.sc, sum_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]

        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.
            max_contributions_per_partition,
            min_value=sum_params.min_value,
            max_value=sum_params.max_value,
            public_partitions=sum_params.public_partitions)
        self.assertEqual(args[1], params)

        self.assertEqual(actual_result.collect(), [(3.0, "pk1")])

    def test_sum_calls_returns_sensible_result(self):
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
        sum_params = agg.SumParams(noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                   max_partitions_contributed=2,
                                   max_contributions_per_partition=3,
                                   min_value=1,
                                   max_value=2,
                                   budget_weight=1,
                                   public_partitions=None,
                                   partition_extractor=lambda x: x[1],
                                   value_extractor=lambda x: x[2])

        # Act
        actual_result = prdd.sum(PrivateRDDTest.sc, sum_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pk1": 90.0}
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
        sum_params = agg.SumParams(noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                   max_partitions_contributed=2,
                                   max_contributions_per_partition=3,
                                   min_value=1,
                                   max_value=2,
                                   budget_weight=1,
                                   partition_extractor=lambda x: x[1],
                                   value_extractor=lambda x: x[2],
                                   public_partitions=["pubK1", "pubK2"])

        # Act
        actual_result = prdd.sum(PrivateRDDTest.sc, sum_params)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        expected_result_dict = {"pubK2": 0.0, "pubK1": 90.0}
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
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([(2,
                                                                      ["pk1"])])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)

        count_params = agg.CountParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            budget_weight=1,
            public_partitions=None,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.count(PrivateRDDTest.sc, count_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]
        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=count_params.max_partitions_contributed,
            max_contributions_per_partition=count_params.
            max_contributions_per_partition,
            public_partitions=count_params.public_partitions)
        self.assertEqual(args[1], params)

        self.assertEqual(actual_result.collect(), [(2, "pk1")])

    def test_count_calls_returns_sensible_result(self):
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

        count_params = agg.CountParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            budget_weight=1,
            public_partitions=None,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.count(PrivateRDDTest.sc, count_params)
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

        count_params = agg.CountParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            budget_weight=1,
            public_partitions=["pubK1", "pubK2"],
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.count(PrivateRDDTest.sc, count_params)
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
        mock_aggregate.return_value = PrivateRDDTest.sc.parallelize([(2,
                                                                      ["pk1"])])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.make_private(dist_data, budget_accountant,
                                          privacy_id_extractor)
        privacy_id_count_params = agg.PrivacyIdCountParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            budget_weight=1,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.privacy_id_count(PrivateRDDTest.sc,
                                              privacy_id_count_params)

        # Assert
        mock_aggregate.assert_called_once()
        args = mock_aggregate.call_args[0]

        rdd = dist_data.map(lambda x: (privacy_id_extractor(x), x))
        self.assertListEqual(args[0].collect(), rdd.collect())

        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            max_partitions_contributed=privacy_id_count_params.
            max_partitions_contributed,
            max_contributions_per_partition=1,
            public_partitions=privacy_id_count_params.public_partitions)
        self.assertEqual(args[1], params)

        self.assertEqual([(2, "pk1")], actual_result.collect())

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
        privacy_id_count_params = agg.PrivacyIdCountParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            budget_weight=1,
            partition_extractor=lambda x: x[1])

        # Act
        actual_result = prdd.privacy_id_count(PrivateRDDTest.sc,
                                              privacy_id_count_params)
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
        privacy_id_count_params = agg.PrivacyIdCountParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            budget_weight=1,
            partition_extractor=lambda x: x[1],
            public_partitions=["pubK1", "pubK2"])

        # Act
        actual_result = prdd.privacy_id_count(PrivateRDDTest.sc,
                                              privacy_id_count_params)
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

        select_partitions_params = agg.SelectPartitionsParams(
            max_partitions_contributed=max_partitions_contributed)
        actual_result = prdd.select_partitions(PrivateRDDTest.sc,
                                               select_partitions_params,
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

        select_partitions_params = agg.SelectPartitionsParams(
            max_partitions_contributed=max_partitions_contributed)
        actual_result = prdd.select_partitions(PrivateRDDTest.sc,
                                               select_partitions_params,
                                               partition_extractor)
        budget_accountant.compute_budgets()

        # Assert
        # This is a health check to validate that the result is sensible.
        # Hence, we use a very large tolerance to reduce test flakiness.
        self.assertEqual(sorted(actual_result.collect()), ["pk1", "pk2"])

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


if __name__ == '__main__':
    unittest.main()
