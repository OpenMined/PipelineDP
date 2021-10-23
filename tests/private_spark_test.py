import pyspark
from pyspark import SparkContext
from unittest.mock import patch
import unittest

import pipeline_dp
from pipeline_dp import aggregate_params as agg
from pipeline_dp import budget_accounting, private_spark


class PrivateRDDTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf()
        cls.sc = SparkContext.getOrCreate(conf=conf)

    def test_map(self):
        data = [(1, 11), (2, 12)]
        dist_data = PrivateRDDTest.sc.parallelize(data)
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.PrivateRDD(dist_data, budget_accountant,
                                        privacy_id_extractor)

        result = prdd.map(lambda x: (x[0], x[1] * 2))

        self.assertEqual(result._rdd.collect(), [(1, 22), (2, 24)])
        self.assertEqual(result._budget_accountant, prdd._budget_accountant)
        self.assertEqual(result._privacy_id_extractor,
                         prdd._privacy_id_extractor)

    def test_flatmap(self):
        data = [(1, 11), (2, 12)]
        dist_data = PrivateRDDTest.sc.parallelize(data)
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return x[0]

        prdd = private_spark.PrivateRDD(dist_data, budget_accountant,
                                        privacy_id_extractor)

        result = prdd.map(lambda x: (x[0], [x[1] * 2, x[1] * 2 + 1]))

        self.assertEqual(result._rdd.collect(),
                         [(1, 22), (1, 23), (2, 24), (2, 25)])
        self.assertEqual(result._budget_accountant, prdd._budget_accountant)
        self.assertEqual(result._privacy_id_extractor,
                         prdd._privacy_id_extractor)

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_sum(self, mock_aggregate):
        dist_data = PrivateRDDTest.sc.parallelize([])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return "pid" + str(x % 10)

        prdd = private_spark.PrivateRDD(dist_data, budget_accountant,
                                        privacy_id_extractor)

        sum_params = agg.SumParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=3,
            low=1,
            high=5,
            budget_weight=1,
            public_partitions=None,
            partition_extractor=lambda x: "pk" + str(x // 10),
            value_extractor=lambda x: x)
        prdd.sum(sum_params)

        mock_aggregate.assert_called_once()

        args = mock_aggregate.call_args[0]

        self.assertEqual(args[0], dist_data)

        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.
            max_contributions_per_partition,
            low=sum_params.low,
            high=sum_params.high,
            public_partitions=sum_params.public_partitions)
        self.assertEqual(args[1], params)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=sum_params.partition_extractor,
            privacy_id_extractor=privacy_id_extractor,
            value_extractor=sum_params.value_extractor)
        self.assertEqual(args[2], data_extractors)

        mock_aggregate.return_value = "some DPEngine.aggregate's return result"
        result = prdd.sum(sum_params)
        self.assertEquals(result, "some DPEngine.aggregate's return result")

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


if __name__ == '__main__':
    unittest.main()
