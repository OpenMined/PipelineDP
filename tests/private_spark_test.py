import pyspark
from unittest.mock import patch
import unittest

import pipeline_dp
from pipeline_dp import aggregate_params as agg
from pipeline_dp import budget_accounting, private_spark


class PrivateRDDTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf()
        cls.sc = pyspark.SparkContext(conf=conf)

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_sum(self, mock_aggregate):
        dist_data = PrivateRDDTest.sc.parallelize([])
        budget_accountant = budget_accounting.NaiveBudgetAccountant(1, 1e-10)

        def privacy_id_extractor(x):
            return "pid" + str(x % 10)

        prdd = private_spark.PrivateRDD(dist_data, budget_accountant, privacy_id_extractor)

        sum_params = agg.SumParams(
            max_partitions_contributed=2, max_contributions_per_partition=3,
            low=1, high=5,
            budget_weight=1, public_partitions=None,
            partition_extractor=lambda x: "pk" + str(x // 10),
            value_extractor=lambda x: x
        )
        prdd.sum(sum_params)

        mock_aggregate.assert_called_once()

        args = mock_aggregate.call_args[0]

        self.assertEqual(args[0], dist_data)

        params = pipeline_dp.AggregateParams(
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.max_contributions_per_partition,
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
