import pyspark
from unittest.mock import patch
import unittest

import pipeline_dp
from pipeline_dp import aggregate_params as agg
from pipeline_dp import accumulator, budget_accounting, dp_computations, private_spark


class PrivateRDDTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf()
        cls.sc = pyspark.SparkContext(conf=conf)

    @patch('pipeline_dp.accumulator.create_accumulator_params')
    def test_sum(self, mock_create_accumulator_params_function):
        data = [11,
                22, 32, 42,  # privacy id 2 is contributing to too many partitions
                65, 65, 65]  # privacy id 5 is contributing too many times to some partition
        dist_data = PrivateRDDTest.sc.parallelize(data)
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
        mock_create_accumulator_params_function.return_value = [
            pipeline_dp.accumulator.AccumulatorParams(
                pipeline_dp.accumulator.SumAccumulator, accumulator.SumParams(
                    noise=dp_computations.MeanVarParams(eps=1,
                                                        delta=1e-10,
                                                        low=1,
                                                        high=3,
                                                        max_partitions_contributed=1,
                                                        max_contributions_per_partition=2,
                                                        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)))]

        # [((pid_1, pk_1), dp_sum_1_1), …, ((pid_m, pk_n), dp_sum_m_n)]
        result = prdd.sum(sum_params).collect()
        keys = {v[0] for v in result}

        self.assertTrue(keys.issuperset({('pid1', 'pk1'), ('pid5', 'pk6')}))

        keys -= {('pid1', 'pk1'), ('pid5', 'pk6')}
        self.assertTrue(keys.issubset({('pid2', 'pk2'), ('pid2', 'pk3'), ('pid2', 'pk4')}))
        self.assertTrue(len(keys), 2)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


if __name__ == '__main__':
    unittest.main()
