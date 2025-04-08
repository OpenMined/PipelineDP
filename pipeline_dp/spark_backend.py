import functools
import random
import numpy as np
from collections.abc import Iterable, Iterator
from typing import Callable, List, Union

import abc
import pipeline_dp.combiners as dp_combiners
import typing
import collections
import itertools
import operator

try:
    import apache_beam as beam
    import apache_beam.transforms.combiners as combiners
except:
    # It is fine if Apache Beam is not installed, other backends can be used.
    pass

class PipelineBackend(abc.ABC):
    """Interface implemented by the pipeline backends compatible with PipelineDP."""

    def to_collection(self, collection_or_iterable, col, stage_name: str):
        """Converts to collection native to Pipeline Framework.

        If collection_or_iterable is already the framework collection then
        its return, if it is iterable, it is converted to the framework
        collection and return.

        Note, that col is required to be the framework collection in order to
        get correct pipeline information.

        Args:
            collection_or_iterable: iterable or Framework collection.
            col: some framework collection.
            stage_name: stage name.
        Returns:
            the framework collection with elements from collection_or_iterable.
        """
        return collection_or_iterable

    def to_multi_transformable_collection(self, col):
        """Converts to a collection, for which multiple transformations can be applied.

        Note: for now it's needed only for LocalBackend, because in Beam and
        Spark any collection can be transformed multiple times.
        """
        return col

    @abc.abstractmethod
    def map(self, col, fn, stage_name: str):
        pass

    @abc.abstractmethod
    def map_with_side_inputs(self, col, fn, side_input_cols: typing.List,
                             stage_name: str):
        """Map with additional (side) inputs for mapper functions.

        Side inputs are passed to fn as arguments. For each `record` fn is
        called with fn(record, side_input1, ..., side_inputn).

        Side input collection, must be singletons (i.e. 1 element collections).
        If you have larger collection, you can apply backend.to_list() to
        make it singleton.

        Args:
            col: framework collection to map.
            fn: callable with arguments (col_record, side_input1, ...,
              side_inputn).
            side_input_cols: list of side inputs (side_input1, ...,
              side_inputn). Side input can be 1 element Python Sequence or
              1 element collections which correspond to the pipeline framework
              (e.g. PCollection for BeamBackend etc). Side inputs will be in
              memory, so they should be small enough.
            stage_name: stage name.
        """

    @abc.abstractmethod
    def flat_map(self, col, fn, stage_name: str):
        """1-to-many map."""

    @abc.abstractmethod
    def flat_map_with_side_inputs(self, col, fn, side_input_cols,
                                  stage_name: str):
        """1-to-many map with side input.

        Side inputs are passed to fn as arguments. For each `record` fn is
        called with fn(record, side_input1, ..., side_inputn).

        Side input collection, must be singletons (i.e. 1 element collections).
        If you have larger collection, you can apply backend.to_list() to
        make it singleton.

       Args:
            col: framework collection to map.
            fn: callable with arguments (col_record, side_input1, ...,
              side_inputn).
            side_input_cols: list of side inputs (side_input1, ...,
              side_inputn). Side input can be 1 element Python Sequence or
              1 element collections which correspond to the pipeline framework
              (e.g. PCollection for BeamBackend etc). Side inputs will be in
              memory, so they should be small enough.
            stage_name: stage name.
        """

    @abc.abstractmethod
    def map_tuple(self, col, fn, stage_name: str):
        pass

    @abc.abstractmethod
    def map_values(self, col, fn, stage_name: str):
        pass

    @abc.abstractmethod
    def group_by_key(self, col, stage_name: str):
        pass

    @abc.abstractmethod
    def filter(self, col, fn, stage_name: str):
        pass

    @abc.abstractmethod
    def filter_by_key(self, col, keys_to_keep, stage_name: str):
        """Filters out elements with keys which are not in `keys_to_keep`.

        Args:
          col: collection with elements (key, data).
          keys_to_keep: collection of keys to keep, both local (currently `list`
            and `set`) and distributed collections are supported.
          stage_name: name of the stage.

        Returns:
          A filtered collection containing only data belonging to keys_to_keep.
        """
        pass

    @abc.abstractmethod
    def keys(self, col, stage_name: str):
        pass

    @abc.abstractmethod
    def values(self, col, stage_name: str):
        pass

    @abc.abstractmethod
    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        """Returns random samples without replacement of values for each key.

        Args:
          col: input collection of elements (key, value).
          n: number of values to sample for each key.
          stage_name: name of the stage.

        Returns:
          A collection of (key, [value]).
        """

    @abc.abstractmethod
    def count_per_element(self, col, stage_name: str):
        pass

    @abc.abstractmethod
    def sum_per_key(self, col, stage_name: str):
        pass

    @abc.abstractmethod
    def combine_accumulators_per_key(self, col, combiner: dp_combiners.Combiner,
                                     stage_name: str):
        """Combines the input collection so that all elements per each key are merged.

        Args:
          col: input collection which contains tuples (key, accumulator).
          combiner: combiner which knows how to perform aggregation on
           accumulators in col.
          stage_name: name of the stage.

        Returns:
          A collection of tuples (key, accumulator).
        """

    @abc.abstractmethod
    def reduce_per_key(self, col, fn: Callable, stage_name: str):
        """Reduces the input collection so that all elements per each key are combined.

        Args:
          col: input collection which contains tuples (key, value).
          fn: function of 2 elements, which returns element of the same
          type. The operation defined by this function needs to be associative
          and commutative (e.g. +, *).
          stage_name: name of the stage.

        Returns:
          A collection of tuples (key, value).
        """

    @abc.abstractmethod
    def flatten(self, cols: Iterable, stage_name: str):
        """Returns a collection that contains all values from collections from cols."""

    @abc.abstractmethod
    def distinct(self, col, stage_name: str):
        """Returns a collection containing distinct elements of the input collection."""

    @abc.abstractmethod
    def to_list(self, col, stage_name: str):
        """Returns a 1-element collection with a list of all elements."""

    def annotate(self, col, stage_name: str, **kwargs):
        """Annotates collection with registered annotators.

        Args:
          col: input collection of elements of any type.
          stage_name: name of the stage.
          kwargs: additional arguments about the current aggregation.

        Returns:
          The input collection after applying annotations from all registered
          annotators.
        """
        return col  # no-op if not implemented for a Backend


class SparkRDDBackend(PipelineBackend):
    """Apache Spark RDD adapter."""

    def __init__(self, sc: 'SparkContext'):
        self._sc = sc

    def to_collection(self, collection_or_iterable, col, stage_name: str):
        # TODO: implement it and remove workaround in map() below.
        return collection_or_iterable

    def map(self, rdd, fn, stage_name: str = None):
        # TODO(make more elegant solution): workaround for public_partitions
        # It is beneficial to accept them as in-memory collection for improving
        # performance of filtering. But for applying map, RDD is required.
        if isinstance(rdd, Iterable):
            return self._sc.parallelize(rdd).map(fn)
        return rdd.map(fn)

    def map_with_side_inputs(self, rdd, fn, side_input_cols, stage_name: str):
        raise NotImplementedError("map_with_side_inputs "
                                  "is not implement in SparkBackend.")

    def flat_map(self, rdd, fn, stage_name: str = None):
        return rdd.flatMap(fn)

    def flat_map_with_side_inputs(self, col, fn, side_input_cols,
                                  stage_name: str):
        raise NotImplementedError("flat_map_with_side_inputs "
                                  "is not implement in SparkBackend.")

    def map_tuple(self, rdd, fn, stage_name: str = None):
        return rdd.map(lambda x: fn(*x))

    def map_values(self, rdd, fn, stage_name: str = None):
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

    def to_list(self, col, stage_name: str):
        raise NotImplementedError("to_list is not implement in SparkBackend.")