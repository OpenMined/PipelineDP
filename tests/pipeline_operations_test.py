import unittest
import pyspark

from pipeline_dp.pipeline_operations import SparkRDDOperations
from pipeline_dp.pipeline_operations import LocalPipelineOperations


class PipelineOperationsTest(unittest.TestCase):
    pass


class SparkRDDOperationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf()
        cls.sc = pyspark.SparkContext(conf=conf)

    def test_sample_fixed_per_key(self):
        spark_operations = SparkRDDOperations()
        data = [(1, 11), (2, 22), (3, 33), (1, 14), (2, 25), (1, 16)]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        rdd = spark_operations.sample_fixed_per_key(dist_data, 2)
        result = dict(rdd.collect())
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(set(result[1]).issubset({11, 14, 16}))
        self.assertSetEqual(set(result[2]), {22, 25})
        self.assertSetEqual(set(result[3]), {33})

    def test_count_per_element(self):
        spark_operations = SparkRDDOperations()
        data = ['a', 'b', 'a']
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        rdd = spark_operations.count_per_element(dist_data)
        result = rdd.collect()
        result = dict(result)
        self.assertDictEqual(result, {'a': 2, 'b': 1})

        @classmethod
        def tearDownClass(cls):
            cls.sc.stop()


class LocalPipelineOperationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = LocalPipelineOperations()

    def test_local_map(self):
        some_map = self.ops.map([1, 2, 3], lambda x: x)
        # some_map is its own consumable iterator
        self.assertIs(some_map, iter(some_map))

        self.assertEqual(list(self.ops.map([1, 2, 3], str)),
                         ["1", "2", "3"])
        self.assertEqual(list(self.ops.map(range(5), lambda x: x ** 2)),
                         [0, 1, 4, 9, 16])

    def test_local_map_tuple(self):
        some_map = self.ops.map([1, 2, 3], lambda x: x)
        # some_map is its own consumable iterator
        self.assertIs(some_map, iter(some_map))

        self.assertEqual(list(self.ops.map([1, 2, 3], str)),
                         ["1", "2", "3"])
        self.assertEqual(list(self.ops.map(range(5), lambda x: x ** 2)),
                         [0, 1, 4, 9, 16])

    def test_local_map_tuple(self):
        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(list(self.ops.map_tuple(tuple_list, lambda k, v: k + v)),
                         [3, 5, 7])

        self.assertEqual(list(self.ops.map_tuple(tuple_list, lambda k, v: (
            str(k), str(v)))), [("1", "2"), ("2", "3"), ("3", "4")])


if __name__ == '__main__':
    unittest.main()
