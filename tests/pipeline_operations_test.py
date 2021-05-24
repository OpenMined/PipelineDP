import unittest
import pyspark

from pipeline_dp import DataExtractors
from pipeline_dp.pipeline_operations import SparkRDDOperations
from pipeline_dp.pipeline_operations import LocalPipelineOperations
from pipeline_dp.pipeline_operations import BeamOperations


class PipelineOperationsTest(unittest.TestCase):
    pass


class BeamOperationsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
      cls.ops = BeamOperations()
      cls.data_extractors = DataExtractors(
          partition_extractor=lambda x: x[1],
          privacy_id_extractor=lambda x: x[0],
          value_extractor=lambda x: x[2])

    def test_filter_partitions_noop(self):
      col = [(1, 6, 1), (2, 7, 1), (3, 6, 1), (4, 7, 1), (5, 8, 1)]
      public_partitions = []
      result = self.ops.filter_partitions(col, public_partitions, self.data_extractors, "Public partition filtering")
      assert result == col


    def test_filter_partitions_remove(self):
      col = [(1, 7, 1), (2, 19, 1), (3, 9, 1), (4, 11, 1), (5, 10, 1)]
      public_partitions = [7, 9]
      result = self.ops.filter_partitions(col, public_partitions, self.data_extractors, "Public partition filtering")
      assert result == [(7, (1, 7, 1)), (9, (3, 9, 1))]


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
        self.assertEqual(list(self.ops.map([], lambda x: x / 0)),
                         [])

        self.assertEqual(list(self.ops.map([1, 2, 3], str)),
                         ["1", "2", "3"])
        self.assertEqual(list(self.ops.map(range(5), lambda x: x ** 2)),
                         [0, 1, 4, 9, 16])

    def test_local_map_tuple(self):
        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(list(self.ops.map_tuple(tuple_list, lambda k, v: k+v)),
                         [3, 5, 7])

        self.assertEqual(list(self.ops.map_tuple(tuple_list, lambda k, v: (
            str(k), str(v)))), [("1", "2"), ("2", "3"), ("3", "4")])

    def test_local_map_values(self):
        self.assertEqual(list(self.ops.map_values([], lambda x: x / 0)),
                         [])

        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(list(self.ops.map_values(tuple_list, str)),
                         [(1, "2"), (2, "3"), (3, "4")])
        self.assertEqual(list(self.ops.map_values(tuple_list, lambda x: x**2)),
                         [(1, 4), (2, 9), (3, 16)])

    def test_local_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.ops.group_by_key(some_dict)), [
                         ("cheese", ["brie", "swiss"]),
                         ("bread", ["sourdough"])])

    def test_local_filter(self):
        self.assertEqual(list(self.ops.filter([], lambda x: True)),
                         [])
        self.assertEqual(list(self.ops.filter([], lambda x: False)),
                         [])

        example_list = [1, 2, 2, 3, 3, 4, 2]

        self.assertEqual(list(self.ops.filter(example_list, lambda x: x % 2)),
                         [1, 3, 3])
        self.assertEqual(list(self.ops.filter(example_list, lambda x: x < 3)),
                         [1, 2, 2, 2])

    def test_local_values(self):
        self.assertEqual(list(self.ops.values([])),
                         [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.ops.values(example_list)),
                         [2, 3, 4, 8])

    def test_local_count_per_element(self):
        example_list = [1, 2, 3, 4, 5, 6, 1, 4, 0, 1]
        result = self.ops.count_per_element(example_list)

        self.assertEqual(dict(result),
                         {1: 3, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 0: 1})

    def test_laziness(self):
        def exceptions_generator_function():
            yield 1 / 0

        def assert_laziness(operator, *args):
            try:
                operator(exceptions_generator_function(), *args)
            except ZeroDivisionError:
               self.fail(f"local {operator.__name__} is not lazy")

        # reading from exceptions_generator_function() results in error:
        self.assertRaises(ZeroDivisionError,
                          next, exceptions_generator_function())

        # lazy operators accept exceptions_generator_function()
        # as argument without raising errors:
        assert_laziness(self.ops.map, str)
        assert_laziness(self.ops.map_values, str)
        assert_laziness(self.ops.filter, bool)
        assert_laziness(self.ops.values)
        assert_laziness(self.ops.count_per_element)


if __name__ == '__main__':
    unittest.main()
