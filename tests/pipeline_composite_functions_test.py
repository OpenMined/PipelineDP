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
from dataclasses import dataclass
from typing import List, Set

import pyspark
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp import pipeline_composite_functions as composite_funcs


def _materialize_col(backend: pipeline_dp.PipelineBackend, col):
    if isinstance(backend, pipeline_dp.SparkRDDBackend):
        return col.collect()
    else:
        return list(col)


def _create_platform_supported_backends(backends_in_scope: Set[str]):
    result = set()
    for backend in backends_in_scope:
        if backend == "local":
            result.add(pipeline_dp.LocalBackend())
        elif backend == "beam":
            result.add(pipeline_dp.BeamBackend())
        elif backend == "spark":
            result.add(
                pipeline_dp.SparkRDDBackend(
                    pyspark.SparkContext.getOrCreate(pyspark.SparkConf())))
        elif backend == "multi_proc_local":
            if sys.platform != 'win32' and sys.platform != 'darwin':
                result.add(
                    pipeline_dp.pipeline_backend.MultiProcLocalBackend(
                        n_jobs=1))
    return result


_ALL_BACKENDS = {"local", "beam", "spark", "multi_proc_local"}


@dataclass
class TestContainer:
    x: int
    y: str
    z: List[str]


class PipelineCompositeFunctionsTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    @parameterized.parameters(_create_platform_supported_backends(_ALL_BACKENDS)
                             )
    def test_key_by_extracts_keys_and_keeps_values_untouched(self, backend):
        col = ["key1_value1", "key1_value2", "key2_value1"]

        def underscore_separated_key_extractor(el):
            return el.split("_")[0]

        result = composite_funcs.key_by(backend,
                                        col,
                                        underscore_separated_key_extractor,
                                        stage_name="Key by")

        self.assertSetEqual(
            {("key1", "key1_value1"), ("key1", "key1_value2"),
             ("key2", "key2_value1")}, set(_materialize_col(backend, result)))

    @parameterized.parameters(_create_platform_supported_backends(_ALL_BACKENDS)
                             )
    def test_size_accounts_duplicates(self, backend):
        col = [3, 2, 1, 1]

        result = composite_funcs.size(backend, col, stage_name="Size")

        self.assertEqual([4], list(_materialize_col(backend, result)))

    @parameterized.parameters(
        _create_platform_supported_backends(_ALL_BACKENDS - {"spark"}))
    def test_collect_to_container_one_element_collections_works(self, backend):
        col_x = [2]
        col_y = ["str"]
        col_z = [["str1", "str2"]]

        container = composite_funcs.collect_to_container(
            backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")

        self.assertEqual([TestContainer(x=2, y="str", z=["str1", "str2"])],
                         list(_materialize_col(backend, container)))

    @parameterized.parameters(
        _create_platform_supported_backends(_ALL_BACKENDS - {"spark"}))
    def test_collect_to_container_collections_with_multiple_elements_preserves_only_one_element(
            self, backend):
        col_x = [2, 1]
        col_y = ["str1", "str2"]
        col_z = [["str1", "str2"], ["str3", "str4"]]

        container = composite_funcs.collect_to_container(
            backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")

        container: TestContainer = list(_materialize_col(backend, container))[0]
        self.assertIn(container.x, col_x)
        self.assertIn(container.y, col_y)
        self.assertIn(container.z, col_z)

    @parameterized.parameters(_create_platform_supported_backends({"spark"}))
    def test_collect_to_container_spark_is_not_supported(self, backend):
        col_x = [2]
        col_y = ["str"]
        col_z = [["str1", "str2"]]

        with self.assertRaises(NotImplementedError):
            composite_funcs.collect_to_container(backend, {
                "x": col_x,
                "y": col_y,
                "z": col_z
            }, TestContainer, "Collect to container")
