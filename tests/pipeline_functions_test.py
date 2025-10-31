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
from dataclasses import dataclass
from typing import List

import apache_beam as beam
import apache_beam.testing.test_pipeline as test_pipeline
import apache_beam.testing.util as beam_util
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp import pipeline_functions as composite_funcs
from pipeline_dp import pipeline_backend


@dataclass
class TestContainer:
    x: int
    y: str
    z: List[str]


class BeamBackendTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = pipeline_dp.BeamBackend()

    def test_key_by_extracts_keys_and_keeps_values_untouched(self):
        with test_pipeline.TestPipeline() as p:
            col = p | beam.Create(["key1_value1", "key1_value2", "key2_value1"])

            def underscore_separated_key_extractor(el):
                return el.split("_")[0]

            result = composite_funcs.key_by(self.backend,
                                            col,
                                            underscore_separated_key_extractor,
                                            stage_name="Key by")

            beam_util.assert_that(
                result,
                beam_util.equal_to({("key1", "key1_value1"),
                                    ("key1", "key1_value2"),
                                    ("key2", "key2_value1")}))

    def test_size_accounts_for_duplicates(self):
        with test_pipeline.TestPipeline() as p:
            col = p | beam.Create([3, 2, 1, 1])

            result = composite_funcs.size(self.backend, col, stage_name="Size")

            beam_util.assert_that(result, beam_util.equal_to([4]))

    def test_collect_to_container_one_element_collections_works(self):
        with test_pipeline.TestPipeline() as p:
            col_x = p | "col_x" >> beam.Create([2])
            col_y = p | "col_y" >> beam.Create(["str"])
            col_z = p | "col_z" >> beam.Create([["str1", "str2"]])

            container = composite_funcs.collect_to_container(
                self.backend, {
                    "x": col_x,
                    "y": col_y,
                    "z": col_z
                }, TestContainer, "Collect to container")

            beam_util.assert_that(
                container,
                beam_util.equal_to(
                    [TestContainer(x=2, y="str", z=["str1", "str2"])]))

    def test_collect_to_container_collections_with_multiple_elements_preserves_only_one_element(
            self):
        # We don't use beam package methods and TestPipeline because in
        # beam_util there is no assertIn method.
        col_x = [2, 1]
        col_y = ["str1", "str2"]
        col_z = [["str1", "str2"], ["str3", "str4"]]

        container = composite_funcs.collect_to_container(
            self.backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")

        container: TestContainer = list(container)[0]
        self.assertIn(container.x, col_x)
        self.assertIn(container.y, col_y)
        self.assertIn(container.z, col_z)

    @parameterized.named_parameters(
        dict(testcase_name='empty collection', col=[], expected_min_max=[]),
        dict(testcase_name='collection with one element',
             col=[("k", 1)],
             expected_min_max=[("k", (1, 1))]),
        dict(testcase_name='collection with more than two elements',
             col=[("a", 1), ("a", 5), ("a", 2), ("b", -1), ("b", 10), ("c", 1)],
             expected_min_max=[("a", (1, 5)), ("b", (-1, 10)), ("c", (1, 1))]))
    def test_min_max_per_key(self, col, expected_min_max):
        with test_pipeline.TestPipeline() as p:
            col = p | beam.Create(col)

            result = composite_funcs.min_max_per_key(self.backend, col,
                                                     "Min and max elements")

            beam_util.assert_that(result, beam_util.equal_to(expected_min_max))


class LocalBackendTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = pipeline_dp.LocalBackend()

    def test_key_by_extracts_keys_and_keeps_values_untouched(self):
        col = ["key1_value1", "key1_value2", "key2_value1"]

        def underscore_separated_key_extractor(el):
            return el.split("_")[0]

        result = composite_funcs.key_by(self.backend,
                                        col,
                                        underscore_separated_key_extractor,
                                        stage_name="Key by")

        self.assertSetEqual(
            {("key1", "key1_value1"), ("key1", "key1_value2"),
             ("key2", "key2_value1")}, set(result))

    def test_size_accounts_for_duplicates(self):
        col = [3, 2, 1, 1]

        result = composite_funcs.size(self.backend, col, stage_name="Size")

        self.assertEqual([4], list(result))

    def test_collect_to_container_one_element_collections_works(self):
        col_x = [2]
        col_y = ["str"]
        col_z = [["str1", "str2"]]

        container = composite_funcs.collect_to_container(
            self.backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")

        self.assertEqual([TestContainer(x=2, y="str", z=["str1", "str2"])],
                         list(container))

    def test_collect_to_container_collections_with_multiple_elements_preserves_only_one_element(
            self):
        col_x = [2, 1]
        col_y = ["str1", "str2"]
        col_z = [["str1", "str2"], ["str3", "str4"]]

        container = composite_funcs.collect_to_container(
            self.backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")

        container: TestContainer = list(container)[0]
        self.assertIn(container.x, col_x)
        self.assertIn(container.y, col_y)
        self.assertIn(container.z, col_z)

    @parameterized.named_parameters(
        dict(testcase_name='empty collection', col=[], expected_min_max=[]),
        dict(testcase_name='collection with one element',
             col=[("k", 1)],
             expected_min_max=[("k", (1, 1))]),
        dict(testcase_name='collection with more than two elements',
             col=[("a", 1), ("a", 5), ("a", 2), ("b", -1), ("b", 10), ("c", 1)],
             expected_min_max=[("a", (1, 5)), ("b", (-1, 10)), ("c", (1, 1))]))
    def test_min_max_per_key(self, col, expected_min_max):
        result = composite_funcs.min_max_per_key(self.backend, col,
                                                 "Min and max elements")

        self.assertEqual(expected_min_max, list(result))

    @parameterized.parameters(1, 5)
    def test_local_filter_by_key(self, sharding_factor):
        col = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10), (7, 4), (7, 5)]
        keys_to_keep = [7, 9]
        result = composite_funcs.filter_by_key_with_sharding(
            self.backend, col, keys_to_keep, sharding_factor, "filter_by_key")
        self.assertEqual(sorted(list(result)), [(7, 1), (7, 4), (7, 5),
                                                (9, 10)])
