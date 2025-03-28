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

import random
from typing import Type, Dict, Any, Callable, TypeVar, Iterable, List

from pipeline_dp import pipeline_backend


def key_by(backend: pipeline_backend.PipelineBackend, col,
           key_extractor: Callable, stage_name: str):
    return backend.map(
        col, lambda el: (key_extractor(el), el),
        f"{stage_name}: key collection by keys from key extractor.")


def size(backend: pipeline_backend.PipelineBackend, col, stage_name: str):
    """Returns a one element collection that contains the size of the input
    collection."""

    col = backend.map(col, lambda x: "fake_common_key",
                      f"{stage_name}: mapping to the same key")
    col = backend.count_per_element(
        col, f"{stage_name}: counting the number of elements")
    return backend.values(col, f"{stage_name}: dropping the fake_common_key")


def collect_to_container(backend: pipeline_backend.PipelineBackend,
                         cols: Dict[str, Any], container_class: Type,
                         stage_name: str):
    """Collects pCollections to one collection containing one element of
    a container class.

    It can be useful if you have different objects in pCollections,
    and you want to store all of them in one container object.

    Important: pCollections in col have to be keyed by the names that are
    exactly the same as the arguments' names in the container class constructor.
    Also, pCollections should contain exactly one element, otherwise behaviour
    is undefined.

    Example:
       @dataclass
       class Container:
           x: int
           y: str
           z: List[str]

       col_x = [2]
       col_y = ["str"]
       col_z = [["str1", "str2"]]
       container = collect({"x": col_x, "y": col_y, "z": col_z}, Container,
                           "stage name")
       # container will be equal to Container(x=2, y="str", z=["str1", "str2"]).

    Args:
      backend: backend to use to perform the computation.
      cols: one element collections to collect in the given container class
        keyed by argument names of the container class constructor.
      container_class: container where to put all the cols, has to be callable,
        i.e. container_class(**args_dict).
      stage_name: name of the stage.

    Returns:
      A collection that contains one instance of the container class.
    """

    def create_key_fn(key):
        # we cannot inline it due to lambda function closures capture
        # (https://stackoverflow.com/questions/2295290/what-do-lambda-function-closures-capture)
        return lambda _: key

    input_list = [
        key_by(backend, col, create_key_fn(key),
               f"{stage_name}: key input cols by their keys")
        for key, col in cols.items()
    ]
    input_list = backend.flatten(
        input_list, f"{stage_name}: input cols to one PCollection")
    input_list = backend.to_list(input_list,
                                 f"{stage_name}: inputs col to one list")
    input_dict = backend.map(
        input_list, dict,
        f"{stage_name}: list of inputs to dictionary of inputs")
    return backend.map(input_dict, lambda d: container_class(**d),
                       f"{stage_name}: construct container class from inputs")


def min_max_per_key(backend: pipeline_backend.PipelineBackend, col,
                    stage_name: str):
    """Returns min and max per key for (key, value) collection."""
    col = backend.map_values(col, lambda x: (x, x),
                             f"{stage_name}: convert x to (x, x).")
    # col: (key, (value, value))

    col = backend.reduce_per_key(
        col,
        lambda x, y: (min(x[0], y[0]), max(x[1], y[1])),
        f"{stage_name}: reduce to compute min, max",
    )
    # col: (key, (min, max))
    return col


def filter_by_key_with_sharding(backend: pipeline_backend.PipelineBackend, col,
                                keys_to_keep, sharding_factor: int,
                                stage_name: str):
    """Filters out elements with keys which are not in `keys_to_keep`.

    It's like `backend.filter_by_key`, but for scalable filtering split each key
    into `sharding_factor` subkeys.

    Args:
      backend: backend to use to perform the computation.
      col: collection with elements (key, data).
      keys_to_keep: collection of keys to keep, both local (currently `list` and
        `set`) and distributed collections are supported.
      sharding_factor: number of subkeys to split each key into.
      stage_name: name of the stage.

     Returns:
      A filtered collection containing only data belonging to keys_to_keep.
    """
    if sharding_factor > 1:
        col = backend.map_tuple(
            col,
            lambda k, v: ((k, random.randint(0, sharding_factor - 1)), v),
            f"Sharding each key into {sharding_factor} subkeys",
        )
        keys_to_keep = backend.flat_map(
            keys_to_keep,
            lambda p: tuple((p, i) for i in range(sharding_factor)),
            f"Shard partitions into {sharding_factor} keys",
        )

    col_filtered = backend.filter_by_key(col, keys_to_keep, stage_name)

    if sharding_factor > 1:
        col_filtered = backend.map_tuple(col_filtered, lambda k, v: (k[0], v),
                                         "Remove sharding factor")

    return col_filtered
