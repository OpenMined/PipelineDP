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
"""Adapters for working with pipeline frameworks."""

import functools
import random
import numpy as np
from collections.abc import Iterable, Iterator
from typing import Callable, List, Optional, Union

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
    def filter_with_side_inputs(self, col, fn, side_input_cols,
                                stage_name: str):
        """Filter with side input.

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


class UniqueLabelsGenerator:
    """Generates unique labels for each pipeline aggregation."""

    def __init__(self, suffix):
        self._labels = set()
        self._suffix = ("_" + suffix) if suffix else ""

    def _add_if_unique(self, label):
        if label in self._labels:
            return False
        self._labels.add(label)
        return True

    def unique(self, label):
        if not label:
            label = "UNDEFINED_STAGE_NAME"
        suffix_label = label + self._suffix
        if self._add_if_unique(suffix_label):
            return suffix_label
        for i in itertools.count(1):
            label_candidate = f"{label}_{i}{self._suffix}"
            if self._add_if_unique(label_candidate):
                return label_candidate


class BeamBackend(PipelineBackend):
    """Apache Beam adapter."""

    def __init__(self, suffix: str = ""):
        super().__init__()
        self._ulg = UniqueLabelsGenerator(suffix)

    @property
    def unique_lable_generator(self) -> UniqueLabelsGenerator:
        return self._ulg

    def to_collection(self, collection_or_iterable, col, stage_name: str):
        if isinstance(collection_or_iterable, beam.PCollection):
            return collection_or_iterable
        return col.pipeline | self._ulg.unique(stage_name) >> beam.Create(
            collection_or_iterable)

    def map(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Map(fn)

    def map_with_side_inputs(self,
                             col,
                             fn,
                             side_input_cols,
                             stage_name: str = None):
        side_inputs = [
            beam.pvalue.AsSingleton(side_input_col)
            for side_input_col in side_input_cols
        ]
        return col | self._ulg.unique(stage_name) >> beam.Map(fn, *side_inputs)

    def flat_map(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.FlatMap(fn)

    def flat_map_with_side_inputs(self, col, fn, side_input_cols,
                                  stage_name: str):
        side_inputs = [
            beam.pvalue.AsSingleton(side_input_col)
            for side_input_col in side_input_cols
        ]
        return col | self._ulg.unique(stage_name) >> beam.FlatMap(
            fn, *side_inputs)

    def map_tuple(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Map(lambda x: fn(*x))

    def map_values(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.MapTuple(lambda k, v:
                                                                   (k, fn(v)))

    def group_by_key(self, col, stage_name: str):
        """Groups the values for each key in the PCollection into a single sequence.

        Args:
          col: input collection with elements (key, value)
          stage_name: name of the stage

        Returns:
          A PCollection of tuples in which the type of the second item is an
          iterable, i.e. (key, Iterable[value]).

        """
        return col | self._ulg.unique(stage_name) >> beam.GroupByKey()

    def filter(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Filter(fn)

    def filter_with_side_inputs(self, col, fn, side_input_cols,
                                stage_name: str):
        side_inputs = [
            beam.pvalue.AsSingleton(side_input_col)
            for side_input_col in side_input_cols
        ]
        return col | self._ulg.unique(stage_name) >> beam.Filter(
            fn, *side_inputs)

    def filter_by_key(self, col, keys_to_keep, stage_name: str):

        class PartitionsFilterJoin(beam.DoFn):

            def process(self, joined_data):
                key, rest = joined_data
                values, to_keep = rest.get(VALUES), rest.get(TO_KEEP)

                if not values:
                    return

                if to_keep:
                    for value in values:
                        yield key, value

        def does_keep(pk_val):
            return pk_val[0] in keys_to_keep

        # define constants for using as keys in CoGroupByKey
        VALUES, TO_KEEP = 0, 1

        if keys_to_keep is None:
            raise TypeError("Must provide a valid keys to keep")

        if isinstance(keys_to_keep, (list, set)):
            # Keys to keep are in memory.
            if not isinstance(keys_to_keep, set):
                keys_to_keep = set(keys_to_keep)
            return col | self._ulg.unique("Filtering out") >> beam.Filter(
                does_keep)

        # `keys_to_keep` are not in memory. Filter out with a join.
        keys_to_keep = (keys_to_keep | self._ulg.unique("Reformat PCollection")
                        >> beam.Map(lambda x: (x, True)))
        return ({
            VALUES: col,
            TO_KEEP: keys_to_keep
        } | self._ulg.unique("CoGroup by values and to_keep partition flag") >>
                beam.CoGroupByKey() | self._ulg.unique("Partitions Filter Join")
                >> beam.ParDo(PartitionsFilterJoin()))

    def keys(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Keys()

    def values(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Values()

    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        return col | self._ulg.unique(
            stage_name) >> combiners.Sample.FixedSizePerKey(n)

    def count_per_element(self, col, stage_name: str):
        return col | self._ulg.unique(
            stage_name) >> combiners.Count.PerElement()

    def sum_per_key(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.CombinePerKey(sum)

    def combine_accumulators_per_key(self, col, combiner: dp_combiners.Combiner,
                                     stage_name: str):

        def merge_accumulators(accumulators):
            res = None
            for acc in accumulators:
                if res:
                    res = combiner.merge_accumulators(res, acc)
                else:
                    res = acc
            return res

        return col | self._ulg.unique(stage_name) >> beam.CombinePerKey(
            merge_accumulators)

    def reduce_per_key(self, col, fn: Callable, stage_name: str):
        combine_fn = lambda elements: functools.reduce(fn, elements)
        return col | self._ulg.unique(stage_name) >> beam.CombinePerKey(
            combine_fn)

    def flatten(self, cols, stage_name: str):
        return cols | self._ulg.unique(stage_name) >> beam.Flatten()

    def distinct(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Distinct()

    def to_list(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.combiners.ToList()

    def annotate(self, col, stage_name: str, **kwargs):
        if not _annotators:
            return col
        for annotator in _annotators:
            col = annotator.annotate(col, self, self._ulg.unique(stage_name),
                                     **kwargs)
        return col


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

    def to_list(self, col, stage_name: str):
        raise NotImplementedError("to_list is not implement in SparkBackend.")


class ReiterableLazyIterable(Iterable):
    """A lazy iterable that can be iterated multiple times.

    It generates elements on the first iteration and stores them.
    Subsequent iterations yield the stored elements.
    """

    def __init__(self, iterable: Iterable):
        """Initializes the ReiterableLazyIterable.

        Args:
            iterable: Iterable to make reiterable
        """
        self._iterable = iterable
        self._cache: List = None
        self._first_run_complete = False

    def __iter__(self) -> Iterator:
        if not self._first_run_complete:
            self._cache = []
            for item in self._iterable:
                self._cache.append(item)
            self._first_run_complete = True
        yield from self._cache


class LazySingleton:
    """Represents a lazily evaluated object expected to resolve to a single value.

    This class accepts either a list, which must contain exactly one element,
    or an iterable, which is expected to yield exactly one element upon
    iteration.

    If initialized with a list, the single element is stored immediately.
    If initialized with an iterable, the iterable is stored, and the element
    is fetched and validated only when the `singleton()` method is first called.
    The fetched element is then cached for subsequent calls.
    """

    def __init__(self, iterable_or_list: Union[List, Iterable]):
        """Initializes the LazySingleton.

        Args:
            iterable_or_list: Either a list containing exactly one element,
                              or an iterable expected to yield one element.

        Raises:
            ValueError: If the input is a list but does not contain exactly
                        one element.
            TypeError: If the input is neither a list nor an instance of
                       collections.abc.Iterable.
        """
        self._singleton = None
        self._iterable = None

        if isinstance(iterable_or_list, list):
            if len(iterable_or_list) != 1:
                raise ValueError(
                    f"Input list must contain exactly one element, but found "
                    f"{len(iterable_or_list)} elements.")
            self._singleton = iterable_or_list[0]
        else:
            if not isinstance(iterable_or_list, Iterable):
                raise TypeError(f"Input must be a list or an Iterable, but got "
                                f"{type(iterable_or_list).__name__}.")
            self._iterable = iterable_or_list

    def singleton(self):
        """Retrieves the single underlying value.

        Returns:
           The single element represented by this instance.

        Raises:
           ValueError: If the instance was initialized with an iterable that
                       yields more than one element.
        """
        if self._singleton:
            return self._singleton
        it = iter(self._iterable)
        self._singleton = next(it)
        try:
            next(it)
        except StopIteration:
            return self._singleton
        raise ValueError("The collection contains more than 1 element.")


class LocalBackend(PipelineBackend):
    """Local Pipeline adapter."""

    def to_multi_transformable_collection(self, col):
        return ReiterableLazyIterable(col)

    def map(self, col, fn, stage_name: typing.Optional[str] = None):
        return map(fn, col)

    def map_with_side_inputs(self,
                             col,
                             fn,
                             side_input_cols,
                             stage_name: str = None):
        side_inputs_singletons = [LazySingleton(s) for s in side_input_cols]

        def map_fn(x):
            side_input_values = [s.singleton() for s in side_inputs_singletons]
            return fn(x, *side_input_values)

        return map(map_fn, col)

    def flat_map(self, col, fn, stage_name: str = None):
        return (x for el in col for x in fn(el))

    def flat_map_with_side_inputs(self, col, fn, side_input_cols,
                                  stage_name: str):
        side_inputs_singletons = [LazySingleton(i) for i in side_input_cols]

        def map_fn(x):
            side_input_values = [s.singleton() for s in side_inputs_singletons]
            return fn(x, *side_input_values)

        return self.flat_map(col, map_fn)

    def map_tuple(self, col, fn, stage_name: str = None):
        return map(lambda x: fn(*x), col)

    def map_values(self, col, fn, stage_name: typing.Optional[str] = None):
        return ((k, fn(v)) for k, v in col)

    def group_by_key(self, col, stage_name: typing.Optional[str] = None):

        def group_by_key_generator():
            d = collections.defaultdict(list)
            for key, value in col:
                d[key].append(value)
            for item in d.items():
                yield item

        return group_by_key_generator()

    def filter(self, col, fn, stage_name: typing.Optional[str] = None):
        return filter(fn, col)

    def filter_with_side_inputs(self,
                                col,
                                fn,
                                side_input_cols,
                                stage_name: Optional[str] = None):
        side_inputs_singletons = [LazySingleton(i) for i in side_input_cols]

        def map_fn(x):
            side_input_values = [s.singleton() for s in side_inputs_singletons]
            return fn(x, *side_input_values)

        return self.filter(col, map_fn)

    def filter_by_key(
        self,
        col,
        keys_to_keep,
        stage_name: typing.Optional[str] = None,
    ):
        if not isinstance(keys_to_keep, set):
            keys_to_keep = set(keys_to_keep)
        return (kv for kv in col if kv[0] in keys_to_keep)

    def keys(self, col, stage_name: typing.Optional[str] = None):
        return (k for k, v in col)

    def values(self, col, stage_name: typing.Optional[str] = None):
        return (v for k, v in col)

    def sample_fixed_per_key(self,
                             col,
                             n: int,
                             stage_name: typing.Optional[str] = None):

        def sample_fixed_per_key_generator():
            for item in self.group_by_key(col):
                key = item[0]
                values = item[1]
                if len(values) > n:
                    sampled_indices = np.random.choice(range(len(values)),
                                                       n,
                                                       replace=False)
                    values = [values[i] for i in sampled_indices]
                yield key, values

        return sample_fixed_per_key_generator()

    def count_per_element(self, col, stage_name: typing.Optional[str] = None):
        yield from collections.Counter(col).items()

    def sum_per_key(self, col, stage_name: typing.Optional[str] = None):
        return self.map_values(self.group_by_key(col), sum)

    def combine_accumulators_per_key(self,
                                     col,
                                     combiner: dp_combiners.Combiner,
                                     stage_name: str = None):

        def merge_accumulators(accumulators):
            return functools.reduce(
                lambda acc1, acc2: combiner.merge_accumulators(acc1, acc2),
                accumulators)

        return self.map_values(self.group_by_key(col), merge_accumulators)

    def reduce_per_key(self, col, fn: Callable, stage_name: str):
        combine_fn = lambda elements: functools.reduce(fn, elements)
        return self.map_values(self.group_by_key(col), combine_fn)

    def flatten(self, cols, stage_name: str = None):
        return itertools.chain(*cols)

    def distinct(self, col, stage_name: str):

        def generator():
            for v in set(col):
                yield v

        return generator()

    def to_list(self, col, stage_name: str):
        return (list(col) for _ in range(1))


class Annotator(abc.ABC):
    """Interface to annotate a PipelineDP pipeline.

    Call register_annotator() with your custom Annotator to annotate your
    pipeline."""

    @abc.abstractmethod
    def annotate(self, col, backend: PipelineBackend, stage_name: str,
                 **kwargs):
        """Annotates a collection.

        Args:
          backend: PipelineBackend of the pipeline.
          stage_name: annotation stage_name, it needs to be correctly propagated
          kwargs: additional arguments about the current aggregation.

        Returns:
            The input collection after applying an annotation.
        """


_annotators = []


def register_annotator(annotator: Annotator):
    _annotators.append(annotator)
