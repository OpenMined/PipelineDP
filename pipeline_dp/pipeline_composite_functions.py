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
"""Pipeline backend operations that are high-level and composite, i.e.
their implementation is framework-agnostic because they use only other primitive
 operations declared in PipelineBackend interface."""

from typing import List, Type

import pipeline_dp


def size(backend: pipeline_dp.PipelineBackend, col, stage_name: str):
    """Returns a one element collection that contains the size of the input
    collection."""

    col = backend.map(col, lambda x: "fake_common_key",
                      f"{stage_name}: mapping to the same key")
    col = backend.count_per_element(
        col, f"{stage_name}: counting the number of elements")
    return backend.values(col, f"{stage_name}: dropping the fake_common_key")


def collect(backend: pipeline_dp.PipelineBackend, cols: List,
            container_class: Type, stage_name: str):
    """Collects pCollections to one collection containing one element of
    a container class.

    It can be useful if you have different objects in pCollections,
    and you want to store all of them in one container object.

    Important: the order of cols has to be the same as the order of
    arguments in the constructor of the container class.

    Example:
       @dataclass
       class Container:
           x: Int
           y: Int

       col_x = [2]
       col_y = [3]
       collect([col_x, col_y], Container)

    Args:
      backend: backend to use to perform the computation.
      cols: collections to collect in the given container class.
      container_class: container where to put all the cols, has to be callable,
        i.e. container_class(*args).
      stage_name: name of the stage.

    Returns:
      A collection that contains one instance of container class.
    """
    input_list = backend.flatten(
        cols, f"{stage_name}: input cols to one PCollection")
    input_list = backend.to_list(input_list,
                                 f"{stage_name}: inputs col to one list")
    return backend.map(input_list, lambda l: container_class(*l),
                       f"{stage_name}: construct container class from inputs")
