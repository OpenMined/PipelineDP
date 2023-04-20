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

import pipeline_dp
from pipeline_dp import pipeline_functions as composite_funcs
from pipeline_dp import pipeline_backend


@dataclass
class TestContainer:
    x: int
    y: str
    z: List[str]


class BeamBackendTest(unittest.TestCase):

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
        # We don't use beam package methods and TestPipeline because in beam_util there is no assertIn method.
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


@unittest.skipIf(sys.version_info.minor <= 7 and sys.version_info.major == 3,
                 "There are some problems with PySpark setup on older python.")
class SparkRDDBackendTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import pyspark
        conf = pyspark.SparkConf()
        cls.sc = pyspark.SparkContext.getOrCreate(conf=conf)
        cls.backend = pipeline_dp.SparkRDDBackend(cls.sc)

    def test_key_by_extracts_keys_and_keeps_values_untouched(self):
        col = self.sc.parallelize(["key1_value1", "key1_value2", "key2_value1"])

        def underscore_separated_key_extractor(el):
            return el.split("_")[0]

        result = composite_funcs.key_by(self.backend,
                                        col,
                                        underscore_separated_key_extractor,
                                        stage_name="Key by").collect()

        self.assertSetEqual(
            {("key1", "key1_value1"), ("key1", "key1_value2"),
             ("key2", "key2_value1")}, set(result))

    def test_size_accounts_for_duplicates(self):
        col = self.sc.parallelize([3, 2, 1, 1])

        result = composite_funcs.size(self.backend, col,
                                      stage_name="Size").collect()

        self.assertEqual([4], result)

    def test_collect_to_container_spark_is_not_supported(self):
        col_x = self.sc.parallelize([2])
        col_y = self.sc.parallelize(["str"])
        col_z = self.sc.parallelize([["str1", "str2"]])

        with self.assertRaises(NotImplementedError):
            composite_funcs.collect_to_container(self.backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")


class LocalBackendTest(unittest.TestCase):

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


@unittest.skipIf(sys.platform == 'win32' or sys.platform == 'darwin',
                 "Problems with serialisation on Windows and macOS")
class MultiProcLocalBackendTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = pipeline_backend.MultiProcLocalBackend(n_jobs=1)

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

    def test_collect_to_container_multi_proc_local_is_not_supported(self):
        col_x = [2]
        col_y = ["str"]
        col_z = [["str1", "str2"]]

        with self.assertRaises(NotImplementedError):
            composite_funcs.collect_to_container(self.backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")
