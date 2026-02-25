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
"""Apache Spark RDD adapter."""
from collections.abc import Iterable
import random
from typing import Callable
import operator

from pipeline_dp import combiners as dp_combiners
from pipeline_dp import pipeline_backend


class SparkRDDBackend(pipeline_backend.PipelineBackend):
    """Apache Spark RDD adapter."""

    def __init__(self, sc: 'SparkContext'):
        self._sc = sc

    def to_collection(self, collection_or_iterable, col, stage_name: str):
        # TODO: implement it and remove workaround in map() below.
        return collection_or_iterable

    def map(self, rdd, fn, stage_name: str = None, resource_hints: dict = None):
        # TODO(make more elegant solution): workaround for public_partitions
        # It is beneficial to accept them as in-memory collection for improving
        # performance of filtering. But for applying map, RDD is required.
        if isinstance(rdd, Iterable):
            return self._sc.parallelize(rdd).map(fn)
        return rdd.map(fn)

    def map_with_side_inputs(self, rdd, fn, side_input_cols, stage_name: str, resource_hints: dict = None):
        raise NotImplementedError("map_with_side_inputs "
                                  "is not implement in SparkBackend.")

    def flat_map(self, rdd, fn, stage_name: str = None, resource_hints: dict = None):
        return rdd.flatMap(fn)

    def flat_map_with_side_inputs(self, col, fn, side_input_cols,
                                  stage_name: str, resource_hints: dict = None):
        raise NotImplementedError("flat_map_with_side_inputs "
                                  "is not implement in SparkBackend.")

    def map_tuple(self, rdd, fn, stage_name: str = None, resource_hints: dict = None):
        return rdd.map(lambda x: fn(*x))

    def map_values(self, rdd, fn, stage_name: str = None, resource_hints: dict = None):
        return rdd.mapValues(fn)

    def group_by_key(self, rdd, stage_name: str = None):
        """Groups the values for each key in the RDD into a single sequence.
        Args:
          rdd: input RDD
          stage_name: not used
        Returns:
          An RDD of tuples in which the type of the second item
          is the pyspark.resultiterable.ResultIterable.
        """
        return rdd.groupByKey()

    def filter(self, rdd, fn, stage_name: str = None):
        return rdd.filter(fn)

    def filter_with_side_inputs(self, rdd, fn, side_input_cols,
                                stage_name: str):
        raise NotImplementedError(
            "filter_with_side_inputs is not implement in SparkBackend.")

    def filter_by_key(self, rdd, keys_to_keep, stage_name: str = None):

        if keys_to_keep is None:
            raise TypeError("Must provide a valid keys to keep")

        if isinstance(keys_to_keep, (list, set)):
            # Keys to keep are local.
            if not isinstance(keys_to_keep, set):
                keys_to_keep = set(keys_to_keep)
            return rdd.filter(lambda x: x[0] in keys_to_keep)

        else:
            filtering_rdd = keys_to_keep.map(lambda x: (x, None))
            return rdd.join(filtering_rdd).map(lambda x: (x[0], x[1][0]))

    def keys(self, rdd, stage_name: str = None):
        return rdd.keys()

    def values(self, rdd, stage_name: str = None):
        return rdd.values()

    def sample_fixed_per_key(self, rdd, n: int, stage_name: str = None):
        """See base class. The sampling is not guaranteed to be uniform."""
        return rdd.mapValues(lambda x: [x]).reduceByKey(
            lambda x, y: random.sample(x + y, min(len(x) + len(y), n)))

    def count_per_element(self, rdd, stage_name: str = None):
        return rdd.map(lambda x: (x, 1)).reduceByKey(operator.add)

    def sum_per_key(self, rdd, stage_name: str = None):
        return rdd.reduceByKey(operator.add)

    def combine_accumulators_per_key(self,
                                     rdd,
                                     combiner: dp_combiners.Combiner,
                                     stage_name: str = None):
        return rdd.reduceByKey(
            lambda acc1, acc2: combiner.merge_accumulators(acc1, acc2))

    def reduce_per_key(self, rdd, fn: Callable, stage_name: str):
        return rdd.reduceByKey(fn)

    def flatten(self, cols, stage_name: str = None):
        return self._sc.union(cols)

    def distinct(self, col, stage_name: str):
        return col.distinct()

    def reshuffle(self, col, stage_name: str):
        raise NotImplementedError("reshuffle is not implement in SparkBackend.")

    def to_list(self, col, stage_name: str):
        raise NotImplementedError("to_list is not implement in SparkBackend.")
