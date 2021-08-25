import unittest
import pyspark

from absl.testing import parameterized
import apache_beam as beam
import apache_beam.testing.test_pipeline as test_pipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
import pytest

from pipeline_dp import DataExtractors
from pipeline_dp.pipeline_operations import MultiProcLocalPipelineOperations, SparkRDDOperations
from pipeline_dp.pipeline_operations import LocalPipelineOperations
from pipeline_dp.pipeline_operations import BeamOperations


class PipelineOperationsTest(unittest.TestCase):
    pass


class BeamOperationsTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ops = BeamOperations()
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    def test_filter_by_key_must_not_be_none(self):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create PCollection" >> beam.Create([(1, 6, 1), (2, 7, 1),
                                                           (3, 6, 1), (4, 7, 1),
                                                           (5, 8, 1)])
            public_partitions = None
            with self.assertRaises(TypeError):
                result = self.ops.filter_by_key(col, public_partitions,
                                                self.data_extractors,
                                                "Public partition filtering")

    @parameterized.parameters(
        {'in_memory': True},
        {'in_memory': False},
    )
    def test_filter_by_key_remove(self, in_memory):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create input data PCollection" >> beam.Create(
                [(1, 7, 1), (2, 19, 1), (3, 9, 1), (4, 11, 1), (5, 10, 1)])
            public_partitions = [7, 9]
            expected_result = [(7, (1, 7, 1)), (9, (3, 9, 1))]
            if not in_memory:
                public_partitions = p | "Create public partitions PCollection" >> beam.Create(
                    public_partitions)
            result = self.ops.filter_by_key(col, public_partitions,
                                            self.data_extractors,
                                            "Public partition filtering")
            assert_that(result, equal_to(expected_result))

    @parameterized.parameters(
        {'in_memory': True},
        {'in_memory': False},
    )
    def test_filter_by_key_pcollection_empty_public_keys(self, in_memory):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create PCollection" >> beam.Create([(1, 6, 1), (2, 7, 1),
                                                           (3, 6, 1), (4, 7, 1),
                                                           (5, 8, 1)])
            public_partitions = []
            expected_result = []
            if not in_memory:
                public_partitions = p | "Create public partitions PCollection" >> beam.Create(
                    public_partitions)
            result = self.ops.filter_by_key(col, public_partitions,
                                            self.data_extractors,
                                            "Public partition filtering")
            assert_that(result, equal_to(expected_result))

    def test_reduce_accumulators_per_key(self):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create PCollection" >> beam.Create([(6, 1), (7, 1),
                                                           (6, 1), (7, 1),
                                                           (8, 1)])
            col = self.ops.map_values(col, SumAccumulator,
                                      "Wrap into accumulators")
            col = self.ops.reduce_accumulators_per_key(col)
            result = col | "Get accumulated values" >> beam.Map(
                lambda row: (row[0], row[1].get_metrics()))

            assert_that(result, equal_to([(6, 2), (7, 2), (8, 1)]))


class SparkRDDOperationsTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        conf = pyspark.SparkConf()
        cls.sc = pyspark.SparkContext(conf=conf)
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    def test_filter_by_key_none_public_partitions(self):
        spark_operations = SparkRDDOperations()
        data = [(1, 11, 111), (2, 22, 222)]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        public_partitions = None
        with self.assertRaises(TypeError):
            spark_operations.filter_by_key(
                dist_data, public_partitions,
                SparkRDDOperationsTest.data_extractors)

    @parameterized.parameters({'distributed': False}, {'distributed': True})
    def test_filter_by_key_empty_public_partitions(self, distributed):
        spark_operations = SparkRDDOperations()
        data = [(1, 11, 111), (2, 22, 222)]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        public_partitions = []
        if distributed:
            public_partitions = SparkRDDOperationsTest.sc.parallelize(
                public_partitions)
        result = spark_operations.filter_by_key(
            dist_data, public_partitions,
            SparkRDDOperationsTest.data_extractors).collect()
        self.assertListEqual(result, [])

    @parameterized.parameters({'distributed': False}, {'distributed': True})
    def test_filter_by_key_nonempty_public_partitions(self, distributed):
        spark_operations = SparkRDDOperations()
        data = [(1, 11, 111), (2, 22, 222)]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        public_partitions = [11, 33]
        if distributed:
            public_partitions = SparkRDDOperationsTest.sc.parallelize(
                public_partitions)
        result = spark_operations.filter_by_key(
            dist_data, public_partitions,
            SparkRDDOperationsTest.data_extractors).collect()
        self.assertListEqual(result, [(11, (1, 11, 111))])

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

    def test_reduce_accumulators_per_key(self):
        spark_operations = SparkRDDOperations()
        data = [(1, 11), (2, 22), (3, 33), (1, 14), (2, 25), (1, 16)]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        rdd = spark_operations.map_values(dist_data, SumAccumulator,
                                          "Wrap into accumulators")
        result = spark_operations\
            .reduce_accumulators_per_key(rdd, "Reduce accumulator per key")\
            .map(lambda row: (row[0], row[1].get_metrics()))\
            .collect()
        result = dict(result)
        self.assertDictEqual(result, {1: 41, 2: 47, 3: 33})

    def test_flat_map(self):
        spark_operations = SparkRDDOperations()
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        self.assertEqual(
            spark_operations.flat_map(dist_data, lambda x: x).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])

        data = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        dist_data = SparkRDDOperationsTest.sc.parallelize(data)
        self.assertEqual(
            spark_operations.flat_map(dist_data, lambda x: x[1]).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            spark_operations.flat_map(
                dist_data,
                lambda x: [(x[0], y) for y in x[1]]).collect(), [("a", 1),
                                                                 ("a", 2),
                                                                 ("a", 3),
                                                                 ("a", 4),
                                                                 ("b", 5),
                                                                 ("b", 6),
                                                                 ("b", 7),
                                                                 ("b", 8)])

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


class LocalPipelineOperationsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ops = LocalPipelineOperations()
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    def test_local_map(self):
        self.assertEqual(list(self.ops.map([], lambda x: x / 0)), [])

        self.assertEqual(list(self.ops.map([1, 2, 3], str)), ["1", "2", "3"])
        self.assertEqual(list(self.ops.map(range(5), lambda x: x**2)),
                         [0, 1, 4, 9, 16])

    def test_local_map_tuple(self):
        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(
            list(self.ops.map_tuple(tuple_list, lambda k, v: k + v)), [3, 5, 7])

        self.assertEqual(
            list(self.ops.map_tuple(tuple_list, lambda k, v: (str(k), str(v)))),
            [("1", "2"), ("2", "3"), ("3", "4")])

    def test_local_map_values(self):
        self.assertEqual(list(self.ops.map_values([], lambda x: x / 0)), [])

        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(list(self.ops.map_values(tuple_list, str)), [(1, "2"),
                                                                      (2, "3"),
                                                                      (3, "4")])
        self.assertEqual(list(self.ops.map_values(tuple_list, lambda x: x**2)),
                         [(1, 4), (2, 9), (3, 16)])

    def test_local_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.ops.group_by_key(some_dict)),
                         [("cheese", ["brie", "swiss"]),
                          ("bread", ["sourdough"])])

    def test_local_filter(self):
        self.assertEqual(list(self.ops.filter([], lambda x: True)), [])
        self.assertEqual(list(self.ops.filter([], lambda x: False)), [])

        example_list = [1, 2, 2, 3, 3, 4, 2]

        self.assertEqual(list(self.ops.filter(example_list, lambda x: x % 2)),
                         [1, 3, 3])
        self.assertEqual(list(self.ops.filter(example_list, lambda x: x < 3)),
                         [1, 2, 2, 2])

    def test_local_filter_by_key_empty_public_keys(self):
        col = [(1, 6, 1), (2, 7, 1), (3, 6, 1), (4, 7, 1), (5, 8, 1)]
        public_partitions = []
        result = self.ops.filter_by_key(col, public_partitions,
                                        self.data_extractors,
                                        "Public partition filtering")
        self.assertEqual(result, [])

    def test_local_filter_by_key_remove(self):
        col = [(1, 7, 1), (2, 19, 1), (3, 9, 1), (4, 11, 1), (5, 10, 1)]
        public_partitions = [7, 9]
        result = self.ops.filter_by_key(col, public_partitions,
                                        self.data_extractors,
                                        "Public partition filtering")
        self.assertEqual(result, [(7, (1, 7, 1)), (9, (3, 9, 1))])

    def test_local_keys(self):
        self.assertEqual(list(self.ops.keys([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.ops.keys(example_list)), [1, 2, 3, 4])

    def test_local_values(self):
        self.assertEqual(list(self.ops.values([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.ops.values(example_list)), [2, 3, 4, 8])

    def test_local_count_per_element(self):
        example_list = [1, 2, 3, 4, 5, 6, 1, 4, 0, 1]
        result = self.ops.count_per_element(example_list)

        self.assertEqual(dict(result), {
            1: 3,
            2: 1,
            3: 1,
            4: 2,
            5: 1,
            6: 1,
            0: 1
        })

    def test_local_reduce_accumulators_per_key(self):
        example_list = [(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)]
        col = self.ops.map_values(example_list, SumAccumulator)
        col = self.ops.reduce_accumulators_per_key(col)
        result = list(map(lambda row: (row[0], row[1].get_metrics()), col))
        self.assertEqual(result, [(1, 6), (2, 4), (3, 8)])

    def test_laziness(self):

        def exceptions_generator_function():
            yield 1 / 0

        def assert_laziness(operator, *args):
            try:
                operator(exceptions_generator_function(), *args)
            except ZeroDivisionError:
                self.fail(f"local {operator.__name__} is not lazy")

        # reading from exceptions_generator_function() results in error:
        self.assertRaises(ZeroDivisionError, next,
                          exceptions_generator_function())

        # lazy operators accept exceptions_generator_function()
        # as argument without raising errors:
        assert_laziness(self.ops.map, str)
        assert_laziness(self.ops.map_values, str)
        assert_laziness(self.ops.filter, bool)
        assert_laziness(self.ops.values)
        assert_laziness(self.ops.keys)
        assert_laziness(self.ops.count_per_element)
        assert_laziness(self.ops.flat_map, str)
        assert_laziness(self.ops.sample_fixed_per_key, int)
        assert_laziness(self.ops.reduce_accumulators_per_key)

    def test_local_sample_fixed_per_key_requires_no_discarding(self):
        input_col = [("pid1", ('pk1', 1)), ("pid1", ('pk2', 1)),
                     ("pid1", ('pk3', 1)), ("pid2", ('pk4', 1))]
        n = 3

        sample_fixed_per_key_result = list(
            self.ops.sample_fixed_per_key(input_col, n))

        expected_result = [("pid1", [('pk1', 1), ('pk2', 1), ('pk3', 1)]),
                           ("pid2", [('pk4', 1)])]
        self.assertEqual(sample_fixed_per_key_result, expected_result)

    def test_local_sample_fixed_per_key_with_sampling(self):
        input_col = [(("pid1", "pk1"), 1), (("pid1", "pk1"), 1),
                     (("pid1", "pk1"), 1), (("pid1", "pk1"), 1),
                     (("pid1", "pk1"), 1), (("pid1", "pk2"), 1),
                     (("pid1", "pk2"), 1)]
        n = 3

        sample_fixed_per_key_result = list(
            self.ops.sample_fixed_per_key(input_col, n))

        self.assertTrue(
            all(
                map(lambda pid_pk_v: len(pid_pk_v[1]) <= n,
                    sample_fixed_per_key_result)))

    def test_local_flat_map(self):
        input_col = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(list(self.ops.flat_map(input_col, lambda x: x)),
                         [1, 2, 3, 4, 5, 6, 7, 8])

        input_col = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        self.assertEqual(list(self.ops.flat_map(input_col, lambda x: x[1])),
                         [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            list(
                self.ops.flat_map(input_col,
                                  lambda x: [(x[0], y) for y in x[1]])),
            [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 5), ("b", 6),
             ("b", 7), ("b", 8)])

    def test_local_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.ops.group_by_key(some_dict)),
                         [("cheese", ["brie", "swiss"]),
                          ("bread", ["sourdough"])])


class MultiProcLocalPipelineOperationsTest(unittest.TestCase):

    @staticmethod
    def partition_extract(x):
        return x[1]

    @staticmethod
    def privacy_id_extract(x):
        return x[0]

    @staticmethod
    def value_extract(x):
        return x[2]

    @classmethod
    def setUpClass(cls):
        cls.ops = MultiProcLocalPipelineOperations(n_jobs=1)
        cls.data_extractors = DataExtractors(
            partition_extractor=cls.partition_extract,
            privacy_id_extractor=cls.privacy_id_extract,
            value_extractor=cls.value_extract)

    def _sort_dataset(self, data):
        if isinstance(data, (str, int, float)):
            return data
        if isinstance(data, tuple):
            return tuple(self._sort_dataset(x) for x in data)
        data = list(data)

        def sortfn(x):
            # make everything a sortable thing!
            if isinstance(x, list):
                return self._sort_dataset(tuple(x))
            return x

        return sorted(data, key=sortfn)

    def assertDatasetsEqual(self, first, second, toplevel=True):
        if toplevel:
            first = self._sort_dataset(first)
            second = self._sort_dataset(second)
        self.assertEqual(type(first), type(second))
        if isinstance(first, (str, int, float)):
            self.assertEqual(first, second)
        elif isinstance(first, tuple):
            self.assertEqual(len(first), len(second))
            if len(first) == 2:
                # key value pair
                k1, v1 = first
                k2, v2 = second
                self.assertEqual(k1, k2)
                self.assertDatasetsEqual(v1, v2, False)
            else:
                self.assertEqual(first, second)
        else:  # all other itearble instances
            first = list(first)
            second = list(second)
            self.assertEqual(len(first), len(second))
            for e1, e2 in zip(first, second):
                self.assertDatasetsEqual(e1, e2, False)

    @pytest.mark.timeout(10)
    def test_multiproc_map(self):
        self.assertDatasetsEqual(list(self.ops.map([], lambda x: x / 0)), [])
        self.assertDatasetsEqual(list(self.ops.map([1, 2, 3], str)),
                                 ["1", "2", "3"])
        self.assertDatasetsEqual(list(self.ops.map(range(5), lambda x: x**2)),
                                 [0, 1, 4, 9, 16])

    @pytest.mark.timeout(10)
    def test_multiproc_map_tuple(self):
        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertDatasetsEqual(
            list(self.ops.map_tuple(tuple_list, lambda k, v: k + v)), [3, 5, 7])

        self.assertDatasetsEqual(
            list(self.ops.map_tuple(tuple_list, lambda k, v: (str(k), str(v)))),
            [("1", "2"), ("2", "3"), ("3", "4")])

    @pytest.mark.timeout(10)
    def test_multiproc_map_values(self):
        self.assertDatasetsEqual(list(self.ops.map_values([], lambda x: x / 0)),
                                 [])

        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertDatasetsEqual(list(self.ops.map_values(tuple_list, str)),
                                 [(1, "2"), (2, "3"), (3, "4")])
        self.assertDatasetsEqual(
            list(self.ops.map_values(tuple_list, lambda x: x**2)), [(1, 4),
                                                                    (2, 9),
                                                                    (3, 16)])

    @pytest.mark.timeout(10)
    def test_multiproc_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertDatasetsEqual(list(self.ops.group_by_key(some_dict)),
                                 [("cheese", ["brie", "swiss"]),
                                  ("bread", ["sourdough"])])

    @pytest.mark.timeout(10)
    def test_multiproc_filter(self):
        self.assertDatasetsEqual(list(self.ops.filter([], lambda x: True)), [])
        self.assertDatasetsEqual(list(self.ops.filter([], lambda x: False)), [])

        example_list = [1, 2, 2, 3, 3, 4, 2]

        self.assertDatasetsEqual(
            list(self.ops.filter(example_list, lambda x: x % 2)), [1, 3, 3])
        self.assertDatasetsEqual(
            list(self.ops.filter(example_list, lambda x: x < 3)), [1, 2, 2, 2])

    @pytest.mark.timeout(10)
    def test_multiproc_filter_by_key_empty_public_keys(self):
        col = [(1, 6, 1), (2, 7, 1), (3, 6, 1), (4, 7, 1), (5, 8, 1)]
        public_partitions = []
        result = self.ops.filter_by_key(col, public_partitions,
                                        self.data_extractors,
                                        "Public partition filtering")
        self.assertEqual(list(result), [])

    @pytest.mark.timeout(10)
    def test_multiproc_filter_by_key_remove(self):
        col = [(1, 7, 1), (2, 19, 1), (3, 9, 1), (4, 11, 1), (5, 10, 1)]
        public_partitions = [7, 9]
        result = self.ops.filter_by_key(col, public_partitions,
                                        self.data_extractors,
                                        "Public partition filtering")
        self.assertDatasetsEqual(list(result), [(7, (1, 7, 1)), (9, (3, 9, 1))])

    @pytest.mark.timeout(10)
    def test_multiproc_keys(self):
        self.assertEqual(list(self.ops.keys([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertDatasetsEqual(list(self.ops.keys(example_list)),
                                 [1, 2, 3, 4])

    @pytest.mark.timeout(10)
    def test_multiproc_values(self):
        self.assertEqual(list(self.ops.values([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertDatasetsEqual(list(self.ops.values(example_list)),
                                 [2, 3, 4, 8])

    @pytest.mark.timeout(10)
    def test_multiproc_count_per_element(self):
        example_list = [1, 2, 3, 4, 5, 6, 1, 4, 0, 1]
        result = dict(self.ops.count_per_element(example_list))

        self.assertDictEqual(result, {1: 3, 2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 0: 1})

    @pytest.mark.timeout(10)
    def test_multiproc_sample_fixed_per_key_requires_no_discarding(self):
        input_col = [("pid1", ('pk1', 1)), ("pid1", ('pk2', 1)),
                     ("pid1", ('pk3', 1)), ("pid2", ('pk4', 1))]
        n = 3

        sample_fixed_per_key_result = list(
            self.ops.sample_fixed_per_key(input_col, n))

        expected_result = [("pid1", [('pk1', 1), ('pk2', 1), ('pk3', 1)]),
                           ("pid2", [('pk4', 1)])]
        self.assertDatasetsEqual(sample_fixed_per_key_result, expected_result)

    @pytest.mark.timeout(10)
    def test_multiproc_sample_fixed_per_key_with_sampling(self):
        input_col = [(("pid1", "pk1"), 1), (("pid1", "pk1"), 1),
                     (("pid1", "pk1"), 1), (("pid1", "pk1"), 1),
                     (("pid1", "pk1"), 1), (("pid1", "pk2"), 1),
                     (("pid1", "pk2"), 1)]
        n = 3

        sample_fixed_per_key_result = list(
            self.ops.sample_fixed_per_key(input_col, n))

        self.assertTrue(
            all(
                map(lambda pid_pk_v: len(pid_pk_v[1]) <= n,
                    sample_fixed_per_key_result)))

    @pytest.mark.timeout(10)
    def test_multiproc_flat_map(self):
        input_col = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertDatasetsEqual(
            list(self.ops.flat_map(input_col, lambda x: x)),
            [1, 2, 3, 4, 5, 6, 7, 8])

        input_col = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        self.assertDatasetsEqual(
            list(self.ops.flat_map(input_col, lambda x: x[1])),
            [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertDatasetsEqual(
            list(
                self.ops.flat_map(input_col,
                                  lambda x: [(x[0], y) for y in x[1]])),
            [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 5), ("b", 6),
             ("b", 7), ("b", 8)])

    # @pytest.mark.timeout(10)
    def test_multiproc_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        result = sorted(self.ops.group_by_key(some_dict))
        result = [(k, sorted(v)) for k, v in result]

        self.assertEqual(result, [
            ("bread", ["sourdough"]),
            ("cheese", ["brie", "swiss"]),
        ])

    def test_laziness(self):

        def exceptions_generator_function():
            yield 1 / 0

        def assert_laziness(operator, *args):
            try:
                operator(exceptions_generator_function(), *args)
            except ZeroDivisionError:
                self.fail(f"local {operator.__name__} is not lazy")

        # reading from exceptions_generator_function() results in error:
        self.assertRaises(ZeroDivisionError, next,
                          exceptions_generator_function())

        # lazy operators accept exceptions_generator_function()
        # as argument without raising errors:
        assert_laziness(self.ops.map, str)
        assert_laziness(self.ops.map_values, str)
        assert_laziness(self.ops.filter, bool)
        assert_laziness(self.ops.values)
        assert_laziness(self.ops.keys)
        assert_laziness(self.ops.count_per_element)
        assert_laziness(self.ops.flat_map, str)
        assert_laziness(self.ops.sample_fixed_per_key, int)


# TODO: Extend the proper Accumulator class once it's available.
class SumAccumulator:
    """A simple accumulator for testing purposes."""

    def __init__(self, v):
        self.sum = v

    def add_value(self, v):
        self.sum += v
        return self

    def get_metrics(self):
        return self.sum

    def add_accumulator(self,
                        accumulator: 'SumAccumulator') -> 'SumAccumulator':
        self.sum += accumulator.sum
        return self


if __name__ == '__main__':
    unittest.main()
