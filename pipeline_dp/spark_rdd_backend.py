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
from typing import Callable, Optional
import operator

try:
    from pyspark import rdd as pyspark_rdd
except ImportError:
    pyspark_rdd = None

from pipeline_dp import combiners as dp_combiners
from pipeline_dp import pipeline_backend

if pyspark_rdd is not None:

    # pylint: disable=too-many-instance-attributes
    class _DeferredParentPipelinedRDD(pyspark_rdd.PipelinedRDD):
        """PipelinedRDD deferring parent _jrdd resolution until evaluated."""

        # pylint: disable=super-init-not-called,invalid-name
        def __init__(self,
                     prev,
                     func,
                     preservesPartitioning: bool = False,
                     isFromBarrier: bool = False):
            # PipelinedRDD.__init__ eagerly touches prev._jrdd or inlines
            # prev.func; initialize state directly and resolve prev._jrdd
            # lazily after compute_budgets() caches the parent RDD.
            if (isinstance(prev, _DeferredParentPipelinedRDD) and
                    prev._is_pipelinable()):
                prev_func = prev.func

                def pipeline_func(split, iterator):
                    return func(split, prev_func(split, iterator))

                self.func = pipeline_func
                self.preservesPartitioning = (prev.preservesPartitioning and
                                              preservesPartitioning)
                self._deferred_prev = prev._deferred_prev
            else:
                self.func = func
                self.preservesPartitioning = preservesPartitioning
                self._deferred_prev = prev

            self._prev_jrdd_deserializer = (
                self._deferred_prev._jrdd_deserializer)
            self.is_cached = False
            self.has_resource_profile = False
            self.is_checkpointed = False
            self.ctx = prev.ctx
            self.prev = prev
            self._jrdd_val = None
            self._id = None
            self._jrdd_deserializer = self.ctx.serializer
            self._bypass_serializer = False
            self.partitioner = (prev.partitioner
                                if self.preservesPartitioning else None)
            self.is_barrier = isFromBarrier or prev._is_barrier()

        @property
        def _prev_jrdd(self):
            return self._deferred_prev._jrdd  # pylint: disable=protected-access

        def getNumPartitions(self) -> int:
            return self._deferred_prev.getNumPartitions()

        def mapPartitionsWithIndex(self,
                                   f,
                                   preservesPartitioning: bool = False):
            return _DeferredParentPipelinedRDD(self, f, preservesPartitioning)


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

    def map_with_side_inputs(self,
                             rdd,
                             fn,
                             side_input_cols,
                             stage_name: str,
                             resource_hints: dict = None):
        raise NotImplementedError("map_with_side_inputs "
                                  "is not implement in SparkBackend.")

    def flat_map(self,
                 rdd,
                 fn,
                 stage_name: str = None,
                 resource_hints: dict = None):
        return rdd.flatMap(fn)

    def flat_map_with_side_inputs(self,
                                  col,
                                  fn,
                                  side_input_cols,
                                  stage_name: str,
                                  resource_hints: dict = None):
        raise NotImplementedError("flat_map_with_side_inputs "
                                  "is not implement in SparkBackend.")

    def map_tuple(self,
                  rdd,
                  fn,
                  stage_name: str = None,
                  resource_hints: dict = None):
        return rdd.map(lambda x: fn(*x))

    def map_values(self,
                   rdd,
                   fn,
                   stage_name: str = None,
                   resource_hints: dict = None):
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

    def cache(self, col):
        """Caches a Spark RDD in memory if not already cached.

        Uses default MEMORY_ONLY caching (`col.cache()`) rather than
        MEMORY_AND_DISK so caller `.cache()` calls remain idempotent instead of
        raising SparkUnsupportedOperationException. Callers needing another
        StorageLevel may call `rdd.unpersist().persist(...)`.
        """
        if not col.is_cached:
            col.cache()
        return col

    def defer_cache(
        self,
        col,
        register_callback: Optional[Callable[[Callable[[], None]],
                                             None]] = None,
    ):
        """Defers caching `col` until `register_callback` runs the cache hook.

        While caching is pending on an uncached PipelinedRDD, narrow
        transforms chained onto `col` use `_DeferredParentPipelinedRDD` so
        PySpark does not inline `col.func` before `col` is cached.
        """
        if col.is_cached:
            return col
        if register_callback is None:
            return self.cache(col)
        if pyspark_rdd is None or not isinstance(col, pyspark_rdd.PipelinedRDD):
            register_callback(lambda: self.cache(col))
            return col

        def deferred_map_partitions(
                f,
                preservesPartitioning: bool = False,  # pylint: disable=invalid-name
        ):
            if col.is_cached:
                return pyspark_rdd.RDD.mapPartitionsWithIndex(
                    col, f, preservesPartitioning)
            return _DeferredParentPipelinedRDD(col, f, preservesPartitioning)

        col.mapPartitionsWithIndex = deferred_map_partitions

        def cache_and_restore():
            col.__dict__.pop("mapPartitionsWithIndex", None)
            self.cache(col)

        register_callback(cache_and_restore)
        return col
