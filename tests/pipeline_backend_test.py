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
import sys
import unittest
from typing import List
from unittest.mock import Mock, MagicMock, patch

import apache_beam as beam
import apache_beam.testing.test_pipeline as test_pipeline
import apache_beam.testing.util as beam_util
import pytest
from absl.testing import parameterized

import pipeline_dp.combiners as dp_combiners
from pipeline_dp import DataExtractors
from pipeline_dp.pipeline_backend import BeamBackend
from pipeline_dp.pipeline_backend import LocalBackend
from pipeline_dp.pipeline_backend import SparkRDDBackend


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
            data = [1, 2]
            col = p | "Create col PCollection" >> beam.Create(data)
            list_side_input = [3, 4, 5]
            list_side_input_col = p | "Create list_side_input PCollection" >>\
                                  beam.Create(
                list_side_input)
            one_element_side_input = [6]
            one_element_side_input_col = p | "Create one_element_side_input " \
                                             "PCollection" >> beam.Create(
                one_element_side_input)
            join_lists_fn = lambda x, l1, l2: [x] + l1 + l2

            result = self.backend.map_with_side_inputs(
                col, join_lists_fn,
                [list_side_input_col, one_element_side_input_col],
                "map_with_side_inputs")

            expected_result = [[1, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
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


class BeamBackendStageNameTest(unittest.TestCase):

    class MockUniqueLabelGenerators:

        def unique(self, stage_name: str = ""):
            return "unique_label"

    @staticmethod
    def _create_mock_pcollection():
        mock = Mock()
        mock.__or__ = MagicMock(return_value=mock)
        return mock

    @staticmethod
    def _test_helper():
        mock_pcollection = BeamBackendStageNameTest._create_mock_pcollection()
        backend = BeamBackend()
        backend._ulg = BeamBackendStageNameTest.MockUniqueLabelGenerators()
        return mock_pcollection, backend

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_map(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.map(mock_pcollection, lambda x: x, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_map_values(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.map_values(mock_pcollection, lambda x: x, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_flat_map(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.flat_map(mock_pcollection, lambda x: x, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_map_tuple(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.map_tuple(mock_pcollection, lambda x: x, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_group_by_key(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.group_by_key(mock_pcollection, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_filter(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.filter(mock_pcollection, lambda x: True, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_filter_by_key(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.filter_by_key(mock_pcollection, [1], "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_keys(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.keys(mock_pcollection, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_values(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.values(mock_pcollection, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_sample_fixed_per_key(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.sample_fixed_per_key(mock_pcollection, 1, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    @patch("apache_beam.transforms.ptransform.PTransform.__rrshift__")
    def test_count_per_element(self, mock_rrshift):
        mock_pcollection, backend = self._test_helper()
        backend.count_per_element(mock_pcollection, "stage_name")
        mock_rrshift.assert_called_once_with("unique_label")

    def test_backend_stage_name_must_be_unique(self):
        backend_1 = BeamBackend("SAME_backend_SUFFIX")
        backend_2 = BeamBackend("SAME_backend_SUFFIX")
        with test_pipeline.TestPipeline() as p:
            col = p | f"UNIQUE_BEAM_CREATE_NAME" >> beam.Create([(6, 1),
                                                                 (6, 2)])
            backend_1.map(col, lambda x: x, "SAME_MAP_NAME")
            with self.assertRaisesRegex(RuntimeError,
                                        expected_regex="A transform with label "
                                        "\"SAME_MAP_NAME_SAME_backend_SUFFIX\" "
                                        "already exists in the"
                                        " pipeline"):
                backend_2.map(col, lambda x: x, "SAME_MAP_NAME")

    def test_one_suffix_multiple_same_stage_name(self):
        backend = BeamBackend("UNIQUE_BACKEND_SUFFIX")
        with test_pipeline.TestPipeline() as p:
            col = p | f"UNIQUE_BEAM_CREATE_NAME" >> beam.Create([(6, 1),
                                                                 (6, 2)])
            backend.map(col, lambda x: x, "SAME_MAP_NAME")
            backend.map(col, lambda x: x, "SAME_MAP_NAME")
            backend.map(col, lambda x: x, "SAME_MAP_NAME")

        self.assertEqual("_UNIQUE_BACKEND_SUFFIX", backend._ulg._suffix)
        self.assertEqual(3, len(backend._ulg._labels))
        self.assertIn("SAME_MAP_NAME_UNIQUE_BACKEND_SUFFIX",
                      backend._ulg._labels)
        self.assertIn("SAME_MAP_NAME_1_UNIQUE_BACKEND_SUFFIX",
                      backend._ulg._labels)
        self.assertIn("SAME_MAP_NAME_2_UNIQUE_BACKEND_SUFFIX",
                      backend._ulg._labels)


@unittest.skipIf(sys.version_info.minor <= 7 and sys.version_info.major == 3,
                 "There are some problems with PySpark setup on older python.")
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


class LocalBackendTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = LocalBackend()
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    def test_local_map(self):
        self.assertEqual(list(self.backend.map([], lambda x: x / 0)), [])

        self.assertEqual(list(self.backend.map([1, 2, 3], str)),
                         ["1", "2", "3"])
        self.assertEqual(list(self.backend.map(range(5), lambda x: x**2)),
                         [0, 1, 4, 9, 16])

    def test_local_map_with_side_inputs(self):
        col = [1, 2]
        # side input must be 1-element iterable, and the single element is a list
        list_side_input_col = [3, 4, 5]
        one_element_side_input_col = [6]
        join_lists_fn = lambda x, l1, l2: [x] + l1 + l2

        result = self.backend.map_with_side_inputs(
            col, join_lists_fn,
            [list_side_input_col, one_element_side_input_col],
            "map_with_side_inputs")

        expected_result = [[1, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
        self.assertEqual(list(result), expected_result)

    def test_local_map_tuple(self):
        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(
            list(self.backend.map_tuple(tuple_list, lambda k, v: k + v)),
            [3, 5, 7])

        self.assertEqual(
            list(
                self.backend.map_tuple(tuple_list, lambda k, v:
                                       (str(k), str(v)))), [("1", "2"),
                                                            ("2", "3"),
                                                            ("3", "4")])

    def test_local_map_values(self):
        self.assertEqual(list(self.backend.map_values([], lambda x: x / 0)), [])

        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(list(self.backend.map_values(tuple_list, str)),
                         [(1, "2"), (2, "3"), (3, "4")])
        self.assertEqual(
            list(self.backend.map_values(tuple_list, lambda x: x**2)),
            [(1, 4), (2, 9), (3, 16)])

    def test_local_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.backend.group_by_key(some_dict)),
                         [("cheese", ["brie", "swiss"]),
                          ("bread", ["sourdough"])])

    def test_local_filter(self):
        self.assertEqual(list(self.backend.filter([], lambda x: True)), [])
        self.assertEqual(list(self.backend.filter([], lambda x: False)), [])

        example_list = [1, 2, 2, 3, 3, 4, 2]

        self.assertEqual(
            list(self.backend.filter(example_list, lambda x: x % 2)), [1, 3, 3])
        self.assertEqual(
            list(self.backend.filter(example_list, lambda x: x < 3)),
            [1, 2, 2, 2])

    def test_local_filter_by_key_empty_keys_to_keep(self):
        col = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10)]
        keys_to_keep = []
        result = self.backend.filter_by_key(col, keys_to_keep, "filter_by_key")
        self.assertEqual(list(result), [])

    def test_local_filter_by_key_remove(self):
        col = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10)]
        keys_to_keep = [7, 9]
        result = self.backend.filter_by_key(col, keys_to_keep, "filter_by_key")
        self.assertEqual(list(result), [(7, 1), (9, 10)])

    def test_local_keys(self):
        self.assertEqual(list(self.backend.keys([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.backend.keys(example_list)), [1, 2, 3, 4])

    def test_local_values(self):
        self.assertEqual(list(self.backend.values([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.backend.values(example_list)), [2, 3, 4, 8])

    def test_local_count_per_element(self):
        example_list = [1, 2, 3, 4, 5, 6, 1, 4, 0, 1]
        result = self.backend.count_per_element(example_list)

        self.assertEqual(dict(result), {
            1: 3,
            2: 1,
            3: 1,
            4: 2,
            5: 1,
            6: 1,
            0: 1
        })

    def test_sum_per_key(self):
        data = [(1, 2), (2, 1), (1, 4), (3, 8), (2, -3), (10, 5)]
        result = list(self.backend.sum_per_key(data))
        self.assertEqual(result, [(1, 6), (2, -2), (3, 8), (10, 5)])

    def test_local_combine_accumulators_per_key(self):
        data = [(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)]
        col = self.backend.group_by_key(data)
        sum_combiner = SumCombiner()
        col = self.backend.map_values(col, sum_combiner.create_accumulator)
        col = self.backend.combine_accumulators_per_key(col, sum_combiner)
        col = self.backend.map_values(col, sum_combiner.compute_metrics)
        result = list(col)
        self.assertEqual(result, [(1, 6), (2, 4), (3, 8)])

    def test_reduce_per_key(self):
        data = [(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)]
        col = self.backend.reduce_per_key(data, lambda x, y: x + y, "Reduce")
        result = list(col)
        self.assertEqual(result, [(1, 6), (2, 4), (3, 8)])

    def test_to_list(self):
        data = [1, 2, 3, 4, 5]
        col = self.backend.to_list(data, "To list")
        result = list(col)
        self.assertEqual(result, [[1, 2, 3, 4, 5]])

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
        assert_laziness(self.backend.map, str)
        assert_laziness(self.backend.map_values, str)
        assert_laziness(self.backend.filter, bool)
        assert_laziness(self.backend.values)
        assert_laziness(self.backend.keys)
        assert_laziness(self.backend.count_per_element)
        assert_laziness(self.backend.sum_per_key)
        assert_laziness(self.backend.flat_map, str)
        assert_laziness(self.backend.sample_fixed_per_key, int)
        assert_laziness(self.backend.filter_by_key, list)
        assert_laziness(self.backend.distinct, str)

    def test_local_sample_fixed_per_key_requires_no_discarding(self):
        input_col = [("pid1", ('pk1', 1)), ("pid1", ('pk2', 1)),
                     ("pid1", ('pk3', 1)), ("pid2", ('pk4', 1))]
        n = 3

        sample_fixed_per_key_result = list(
            self.backend.sample_fixed_per_key(input_col, n))

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
            self.backend.sample_fixed_per_key(input_col, n))

        self.assertTrue(
            all(
                map(lambda pid_pk_v: len(pid_pk_v[1]) <= n,
                    sample_fixed_per_key_result)))

    def test_local_flat_map(self):
        input_col = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(list(self.backend.flat_map(input_col, lambda x: x)),
                         [1, 2, 3, 4, 5, 6, 7, 8])

        input_col = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        self.assertEqual(list(self.backend.flat_map(input_col, lambda x: x[1])),
                         [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            list(
                self.backend.flat_map(input_col,
                                      lambda x: [(x[0], y) for y in x[1]])),
            [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 5), ("b", 6),
             ("b", 7), ("b", 8)])

    def test_local_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.backend.group_by_key(some_dict)),
                         [("cheese", ["brie", "swiss"]),
                          ("bread", ["sourdough"])])

    def test_flatten(self):
        data1, data2, data3 = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10]

        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         list(self.backend.flatten((data1, data2, data3))))

    def test_distinct(self):
        input = [3, 2, 1, 3, 5, 4, 1, 1, 2]
        output = set(self.backend.distinct(input, "distinct"))
        self.assertSetEqual({1, 2, 3, 4, 5}, output)


class SumCombiner(dp_combiners.Combiner):

    def create_accumulator(self, values) -> float:
        return sum(values)

    def merge_accumulators(self, sum1: float, sum2: float):
        return sum1 + sum2

    def compute_metrics(self, sum: float) -> float:
        return sum

    def metrics_names(self) -> List[str]:
        return ['sum']

    def explain_computation(self) -> str:
        return "Compute non-dp Sum for tests"


if __name__ == '__main__':
    unittest.main()
