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
import unittest
from unittest.mock import Mock, MagicMock, patch

import apache_beam as beam
import apache_beam.testing.test_pipeline as test_pipeline
import apache_beam.testing.util as beam_util
from absl.testing import parameterized

from pipeline_dp import DataExtractors
from pipeline_dp.beam_backend import BeamBackend
import pipeline_dp.combiners as dp_combiners
from typing import List


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

    def test_filter_with_side_inputs(self):
        with test_pipeline.TestPipeline() as p:
            data = p | "Create data" >> beam.Create([1, 2, 3, 4, 5, 6])
            side_input1 = p | "Create side_input1" >> beam.Create([[2, 4]])
            side_input2 = p | "Create side_input2" >> beam.Create([[5]])

            def filter_fn(x, side_input1, side_input2):
                return x in side_input1 or x in side_input2

            result = self.backend.filter_with_side_inputs(
                data, filter_fn, [side_input1, side_input2], "Filter")

            beam_util.assert_that(result, beam_util.equal_to([2, 4, 5]))

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
