import unittest
import pyspark

import pipeline_dp


class PipelineOperationsTest(unittest.TestCase):
  pass


class SparkOperationsTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    conf = pyspark.SparkConf()
    SparkOperationsTest.sc = pyspark.SparkContext(conf=conf)

  def test_sample_fixed_per_key(self):
    spark_operations = pipeline_dp.SparkOperations()
    data = [(1, 11), (2, 22), (3, 33), (1, 14), (2, 25), (1, 16)]
    dist_data = SparkOperationsTest.sc.parallelize(data)
    rdd = spark_operations.sample_fixed_per_key(dist_data, 2)
    result = dict(rdd.collect())
    self.assertEqual(len(result[1]), 2)
    self.assertTrue(set(result[1]).issubset({11, 14, 16}))
    self.assertSetEqual(set(result[2]), {22, 25})
    self.assertSetEqual(set(result[3]), {33})

  def test_count_per_element(self):
    spark_operations = pipeline_dp.SparkOperations()
    data = ['a', 'b', 'a']
    dist_data = SparkOperationsTest.sc.parallelize(data)
    rdd = spark_operations.count_per_element(dist_data)
    result = rdd.collect()
    result = dict(result)
    self.assertDictEqual(result, {'a': 2, 'b': 1})

    @classmethod
    def tearDownClass(cls):
      SparkOperationsTest.sc.stop()


class LocalPipelineOperationsTest(unittest.TestCase):
  pass


if __name__ == '__main__':
  unittest.main()
