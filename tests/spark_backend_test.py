import sys
import unittest
from typing import Iterable, List
from unittest.mock import Mock, MagicMock, patch

import apache_beam as beam
import apache_beam.testing.test_pipeline as test_pipeline
import apache_beam.testing.util as beam_util
from absl.testing import parameterized

import pipeline_dp.combiners as dp_combiners
from pipeline_dp import DataExtractors
from pipeline_dp.pipeline_backend import BeamBackend
from pipeline_dp.pipeline_backend import LocalBackend, LazySingleton
from pipeline_dp.spark_backend import SparkRDDBackend



class BeamBackendTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = BeamBackend()
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    @parameterized.parameters(True, False)
    def test_to_collection(self, input_is_pcollection):
        with test_pipeline.TestPipeline() as p:
            input = [1, 3, 5]
            if input_is_pcollection:
                input = p | "Creat input PCollection" >> beam.Create(input)
            col = p | beam.Create([])
            output = self.backend.to_collection(input, col, "to_collection")
            self.assertIsInstance(output, beam.PCollection)
            beam_util.assert_that(output, beam_util.equal_to([1, 3, 5]))

    def test_map_with_side_inputs(self):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create data" >> beam.Create([1, 2])
            side_input1 = p | "Create side_int1" >> beam.Create([5])
            side_input2 = p | "Create side_int2" >> beam.Create([30])
            add_fn = lambda x, s1, s2: x + s1 + s2

            result = self.backend.map_with_side_inputs(
                col, add_fn, [side_input1, side_input2], "map_with_side_inputs")

            expected_result = [36, 37]
            beam_util.assert_that(result, beam_util.equal_to(expected_result))

    def test_flat_map_with_side_inputs(self):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create data" >> beam.Create([[1, 2], [3]])
            side_input1 = p | "Create side_int1" >> beam.Create([5])
            side_input2 = p | "Create side_int2" >> beam.Create([30])

            def add_fn(elems, s1, s2):
                for elem in elems:
                    yield elem + s1 + s2

            result = self.backend.flat_map_with_side_inputs(
                col, add_fn, [side_input1, side_input2], "map_with_side_inputs")

            expected_result = [36, 37, 38]
            beam_util.assert_that(result, beam_util.equal_to(expected_result))

    def test_filter_by_key_must_not_be_none(self):
        with test_pipeline.TestPipeline() as p:
            data = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10)]
            col = p | "Create PCollection" >> beam.Create(data)
            key_to_keep = None
            with self.assertRaises(TypeError):
                self.backend.filter_by_key(col, key_to_keep, "filter_by_key")

    @parameterized.parameters(
        {'in_memory': True},
        {'in_memory': False},
    )
    def test_filter_by_key_remove(self, in_memory):
        with test_pipeline.TestPipeline() as p:
            data = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10)]
            col = p | "Create PCollection" >> beam.Create(data)
            keys_to_keep = [7, 9, 9]
            expected_result = [(7, 1), (9, 10)]
            if not in_memory:
                keys_to_keep = p | "To PCollection" >> beam.Create(keys_to_keep)
            result = self.backend.filter_by_key(col, keys_to_keep,
                                                "filter_by_key")
            beam_util.assert_that(result, beam_util.equal_to(expected_result))

    @parameterized.parameters(
        {'in_memory': True},
        {'in_memory': False},
    )
    def test_filter_by_key_empty_keys_to_keep(self, in_memory):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create PCollection" >> beam.Create([(7, 1), (2, 1),
                                                           (3, 9), (4, 1),
                                                           (9, 10)])
            keys_to_keep = []
            if not in_memory:
                keys_to_keep = p | "To PCollection" >> beam.Create(keys_to_keep)
            result = self.backend.filter_by_key(col, keys_to_keep,
                                                "filter_by_key")
            beam_util.assert_that(result, beam_util.equal_to([]))

    def test_combine_accumulators_per_key(self):
        with test_pipeline.TestPipeline() as p:
            col = p | "Create PCollection" >> beam.Create([(6, 1), (7, 1),
                                                           (6, 1), (7, 1),
                                                           (8, 1)])
            sum_combiner = SumCombiner()
            col = self.backend.group_by_key(col, "group_by_key")
            col = self.backend.map_values(col, sum_combiner.create_accumulator,
                                          "Wrap into accumulators")
            col = self.backend.combine_accumulators_per_key(
                col, sum_combiner, "Reduce accumulators per key")
            result = self.backend.map_values(col, sum_combiner.compute_metrics,
                                             "Compute metrics")

            beam_util.assert_that(result,
                                  beam_util.equal_to([(6, 2), (7, 2), (8, 1)]))

    def test_local_combine_accumulators_per_key(self):
        with test_pipeline.TestPipeline() as p:
            data = p | beam.Create([(1, 2), (1, 5), (2, 1), (1, 4), (3, 8),
                                    (2, 3)])
            col = self.backend.reduce_per_key(data, lambda x, y: x + y,
                                              "Reduce")
            beam_util.assert_that(col,
                                  beam_util.equal_to([(1, 11), (2, 4), (3, 8)]))

    def test_to_list(self):
        with test_pipeline.TestPipeline() as p:
            data = p | beam.Create([1, 2, 3, 4, 5])
            col = self.backend.to_list(data, "To list")
            beam_util.assert_that(col, beam_util.equal_to([[1, 2, 3, 4, 5]]))

    def test_flatten(self):
        with test_pipeline.TestPipeline() as p:
            data1 = p | "data1" >> beam.Create([1, 2, 3, 4])
            data2 = p | "data2" >> beam.Create([5, 6, 7, 8])
            col = self.backend.flatten((data1, data2), "flatten")
            beam_util.assert_that(col,
                                  beam_util.equal_to([1, 2, 3, 4, 5, 6, 7, 8]))

    def test_distinct(self):
        with test_pipeline.TestPipeline() as p:
            input = p | beam.Create([3, 2, 1, 3, 5, 4, 1, 1, 2])
            output = self.backend.distinct(input, "distinct")
            beam_util.assert_that(output, beam_util.equal_to([1, 2, 3, 4, 5]))

    def test_sum_per_key(self):
        with test_pipeline.TestPipeline() as p:
            data = p | beam.Create([(1, 2), (2, 1), (1, 4), (3, 8), (2, -3),
                                    (10, 5)])
            result = self.backend.sum_per_key(data, "sum_per_key")
            beam_util.assert_that(
                result, beam_util.equal_to([(1, 6), (2, -2), (3, 8), (10, 5)]))





class SparkRDDBackendTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyspark
        conf = pyspark.SparkConf()
        cls.sc = pyspark.SparkContext.getOrCreate(conf=conf)
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])
        cls.backend = SparkRDDBackend(cls.sc)

    def test_filter_by_key_none_keys_to_keep(self):
        data = [(1, 11), (2, 22)]
        dist_data = self.sc.parallelize(data)
        key_to_keep = None
        with self.assertRaises(TypeError):
            self.backend.filter_by_key(dist_data, key_to_keep)

    @parameterized.parameters({'distributed': False}, {'distributed': True})
    def test_filter_by_key_empty_keys_to_keep(self, distributed):
        data = [(1, 11), (2, 22)]
        dist_data = self.sc.parallelize(data)
        keys_to_keep = []
        if distributed:
            keys_to_keep = self.sc.parallelize(keys_to_keep)
        result = self.backend.filter_by_key(dist_data, keys_to_keep).collect()
        self.assertListEqual(result, [])

    @parameterized.parameters({'distributed': False}, {'distributed': True})
    def test_filter_by_key_nonempty_keys_to_keep(self, distributed):
        data = [(1, 11), (2, 22)]
        dist_data = self.sc.parallelize(data)
        keys_to_keep = [1, 3, 3]
        if distributed:
            keys_to_keep = self.sc.parallelize(keys_to_keep)
        result = self.backend.filter_by_key(dist_data, keys_to_keep).collect()
        self.assertListEqual(result, [(1, 11)])

    def test_sample_fixed_per_key(self):
        data = [(1, 11), (2, 22), (3, 33), (1, 14), (2, 25), (1, 16)]
        dist_data = self.sc.parallelize(data)
        rdd = self.backend.sample_fixed_per_key(dist_data, 2)
        result = dict(rdd.collect())
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(set(result[1]).issubset({11, 14, 16}))
        self.assertSetEqual(set(result[2]), {22, 25})
        self.assertSetEqual(set(result[3]), {33})

    def test_count_per_element(self):
        data = ['a', 'b', 'a']
        dist_data = self.sc.parallelize(data)
        rdd = self.backend.count_per_element(dist_data)
        result = rdd.collect()
        result = dict(result)
        self.assertDictEqual(result, {'a': 2, 'b': 1})

    def test_sum_per_key(self):
        data = self.sc.parallelize([(1, 2), (2, 1), (1, 4), (3, 8), (2, -3),
                                    (10, 5)])
        result = self.backend.sum_per_key(data).collect()
        self.assertEqual(set(result), {(1, 6), (2, -2), (3, 8), (10, 5)})

    @unittest.skipIf(sys.platform == "darwin",
                     "There are some problems with PySpark setup on macOS")
    def test_combine_accumulators_per_key(self):
        data = self.sc.parallelize([(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)])
        rdd = self.backend.group_by_key(data)
        sum_combiner = SumCombiner()
        rdd = self.backend.map_values(rdd, sum_combiner.create_accumulator)
        rdd = self.backend.combine_accumulators_per_key(rdd, sum_combiner)
        rdd = self.backend.map_values(rdd, sum_combiner.compute_metrics)
        result = dict(rdd.collect())
        self.assertDictEqual(result, {1: 6, 2: 4, 3: 8})

    def test_map_tuple(self):
        data = [(1, 2), (3, 4)]
        dist_data = self.sc.parallelize(data)
        result = self.backend.map_tuple(dist_data, lambda a, b: a + b).collect()
        self.assertEqual(result, [3, 7])

    def test_map_with_side_inputs(self):
        with self.assertRaises(NotImplementedError):
            self.backend.map_with_side_inputs(None, None, None,
                                              "map_with_side_inputs")

    def test_flat_map(self):
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        dist_data = self.sc.parallelize(data)
        self.assertEqual(
            self.backend.flat_map(dist_data, lambda x: x).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])

        data = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        dist_data = self.sc.parallelize(data)
        self.assertEqual(
            self.backend.flat_map(dist_data, lambda x: x[1]).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            self.backend.flat_map(
                dist_data,
                lambda x: [(x[0], y) for y in x[1]]).collect(), [("a", 1),
                                                                 ("a", 2),
                                                                 ("a", 3),
                                                                 ("a", 4),
                                                                 ("b", 5),
                                                                 ("b", 6),
                                                                 ("b", 7),
                                                                 ("b", 8)])

    def test_flatten(self):
        data1 = self.sc.parallelize([1, 2, 3, 4])
        data2 = self.sc.parallelize([5, 6, 7, 8])

        self.assertEqual(
            self.backend.flatten((data1, data2)).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])

    def test_distinct(self):
        input = self.sc.parallelize([3, 2, 1, 3, 5, 4, 1, 1, 2])
        output = self.backend.distinct(input, "distinct").collect()
        self.assertSetEqual({1, 2, 3, 4, 5}, set(output))

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()