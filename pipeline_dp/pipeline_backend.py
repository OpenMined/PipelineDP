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
import multiprocessing as mp
import random
import numpy as np
from collections.abc import Iterable
from typing import Callable

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
    def flat_map(self, col, fn, stage_name: str):
        pass

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
        pass

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
        self._suffix = suffix

    def _add_if_unique(self, label):
        if label in self._labels:
            return False
        self._labels.add(label)
        return True

    def unique(self, label):
        if not label:
            label = "UNDEFINED_STAGE_NAME"
        suffix_label = label + "_" + self._suffix
        if self._add_if_unique(suffix_label):
            return suffix_label
        for i in itertools.count(1):
            label_candidate = f"{label}_{i}_{self._suffix}"
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

    def flat_map(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.FlatMap(fn)

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
            col = annotator.annotate(col, self._ulg.unique(stage_name),
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

    def flat_map(self, rdd, fn, stage_name: str = None):
        return rdd.flatMap(fn)

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


class LocalBackend(PipelineBackend):
    """Local Pipeline adapter."""

    def to_multi_transformable_collection(self, col):
        return list(col)

    def map(self, col, fn, stage_name: typing.Optional[str] = None):
        return map(fn, col)

    def flat_map(self, col, fn, stage_name: str = None):
        return (x for el in col for x in fn(el))

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

    def filter_by_key(
        self,
        col,
        keys_to_keep,
        stage_name: typing.Optional[str] = None,
    ):
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


# workaround for passing lambda functions to multiprocessing
# according to https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
_pool_current_func = None


def _pool_worker_init(func):
    global _pool_current_func
    _pool_current_func = func


def _pool_worker(row):
    return _pool_current_func(row)


class _LazyMultiProcIterator:

    def __init__(self, job: typing.Callable, job_inputs: typing.Iterable,
                 chunksize: int, n_jobs: typing.Optional[int], **pool_kwargs):
        """Utilizes the `multiprocessing.Pool.map` for distributed execution of 
        a function `job` on an iterable `job_inputs`.

        Args:
            job: the function to be called on each input

            job_inputs: iterable containing all the inputs

            chunksize: see [multiprocessing.Pool.map signature](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map).  

            n_jobs: see [multiprocessing.Pool constructor](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool) arguments
        """
        self.job = job
        self.chunksize = chunksize
        self.job_inputs = job_inputs
        self.n_jobs = n_jobs
        self.pool_kwargs = pool_kwargs
        self._outputs = None  # type: typing.Optional[typing.Iterator]
        self._pool = None

    def _init_pool(self):
        """Creates the multiprocessing.Pool object that will manage the distributed computation."""
        self._pool = mp.Pool(self.n_jobs,
                             initializer=_pool_worker_init,
                             initargs=(self.job,),
                             **self.pool_kwargs)
        return self._pool

    def _trigger_iterations(self):
        """Trigger the Pool operation that iterates over inputs and produces outputs."""
        if self._outputs is None:
            self._outputs = self._init_pool().map(_pool_worker, self.job_inputs,
                                                  self.chunksize)

    def __iter__(self):
        if isinstance(self.job_inputs, _LazyMultiProcIterator):
            self.job_inputs._trigger_iterations()
        self._trigger_iterations()
        yield from self._outputs


class _LazyMultiProcGroupByIterator(_LazyMultiProcIterator):

    def __init__(self, job_inputs: typing.Iterable, chunksize: int,
                 n_jobs: typing.Optional[int], **pool_kwargs):
        """Utilizes mp.Pool for distributed group by computation.
        The results are held in a `mp.Manager.dict[KeyType, np.Manager.list[ValueType]]`.

        The `mp.Manager.{dict, list}` objects are managed by the `manager` to allow multiprocess-safe
        access to the containers.
        """
        self.manager = mp.Manager()
        self.results_dict = self.manager.dict()

        def insert_row(captures, row):
            (results_dict_,) = captures
            key, val = row
            results_dict_[key].append(val)

        insert_row = functools.partial(insert_row, (self.results_dict,))

        super().__init__(insert_row,
                         job_inputs,
                         chunksize=chunksize,
                         n_jobs=n_jobs,
                         **pool_kwargs)

    def _trigger_iterations(self):
        if self._outputs is None:
            keys = set(k for k, v in self.job_inputs)
            self.results_dict.update({k: self.manager.list() for k in keys})
            self._init_pool().map(_pool_worker, self.job_inputs, self.chunksize)
            self._outputs = ((k, list(v)) for k, v in self.results_dict.items())


class _LazyMultiProcCountIterator(_LazyMultiProcIterator):

    def __init__(self, job_inputs: typing.Iterable, chunksize: int,
                 n_jobs: typing.Optional[int], **pool_kwargs):
        """Utilizes mp.Pool for distributed group by computation.
        The results are held in a `mp.Manager.dict[KeyType, int]`.

        The `mp.Manager.dict` object is managed by the `manager` to allow multiprocess-safe
        access to the container.
        """
        self.manager = mp.Manager()
        self.results_dict = self.manager.dict()

        def insert_row(captures, key):
            (results_dict_,) = captures
            results_dict_[key] += 1

        insert_row = functools.partial(insert_row, (self.results_dict,))

        super().__init__(insert_row,
                         job_inputs,
                         chunksize=chunksize,
                         n_jobs=n_jobs,
                         **pool_kwargs)

    def _trigger_iterations(self):
        if self._outputs is None:
            keys = set(self.job_inputs)
            self.results_dict.update({k: 0 for k in keys})
            self._init_pool().map(_pool_worker, self.job_inputs, self.chunksize)
            self._outputs = self.results_dict.items()


class MultiProcLocalBackend(PipelineBackend):
    """Warning: this class is experimental."""

    def __init__(self,
                 n_jobs: typing.Optional[int] = None,
                 chunksize: int = 1,
                 **pool_kwargs):
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.pool_kwargs = pool_kwargs

    def map(self, col, fn, stage_name: typing.Optional[str] = None):
        return _LazyMultiProcIterator(job=fn,
                                      job_inputs=col,
                                      n_jobs=self.n_jobs,
                                      chunksize=self.chunksize,
                                      **self.pool_kwargs)

    def flat_map(self, col, fn, stage_name: typing.Optional[str] = None):
        return (e for x in self.map(col, fn, stage_name) for e in x)

    def map_tuple(self, col, fn, stage_name: typing.Optional[str] = None):
        return self.map(col, lambda row: fn(*row), stage_name)

    def map_values(self, col, fn, stage_name: typing.Optional[str] = None):
        return self.map(col, lambda x: (x[0], fn(x[1])), stage_name)

    def group_by_key(self, col, stage_name: typing.Optional[str] = None):
        return _LazyMultiProcGroupByIterator(col, self.chunksize, self.n_jobs,
                                             **self.pool_kwargs)

    def filter(self, col, fn, stage_name: typing.Optional[str] = None):
        ordered_predicates = self.map(col, fn, stage_name)
        return (row for row, keep in zip(col, ordered_predicates) if keep)

    def filter_by_key(self,
                      col,
                      keys_to_keep,
                      stage_name: typing.Optional[str] = None):

        def mapped_fn(keys_to_keep_, kv):
            return kv, (kv[0] in keys_to_keep_)

        mapped_fn = functools.partial(mapped_fn, keys_to_keep)
        key_keep = self.map(col, mapped_fn, stage_name)
        return (row for row, keep in key_keep if keep)

    def keys(self, col, stage_name: typing.Optional[str] = None):
        # no point in passing through multiproc.
        return (k for k, v in col)

    def values(self, col, stage_name: typing.Optional[str] = None):
        # no point in passing through multiproc.
        return (v for k, v in col)

    def sample_fixed_per_key(self,
                             col,
                             n: int,
                             stage_name: typing.Optional[str] = None):

        def mapped_fn(captures, row):
            (n_,) = captures
            partition_key, values = row
            samples = values
            if len(samples) > n_:
                samples = random.sample(samples, n_)
            return partition_key, samples

        mapped_fn = functools.partial(mapped_fn, (n,))
        groups = self.group_by_key(col, stage_name)
        return self.map(groups, mapped_fn, stage_name)

    def count_per_element(self, col, stage_name: typing.Optional[str] = None):
        return _LazyMultiProcCountIterator(col, self.chunksize, self.n_jobs,
                                           **self.pool_kwargs)

    def sum_per_key(self, rdd, stage_name: str = None):
        raise NotImplementedError(
            "sum_per_key is not implemented for MultiProcLocalBackend")

    def combine_accumulators_per_key(self, col, combiner: dp_combiners.Combiner,
                                     stage_name: str):
        raise NotImplementedError(
            "combine_accumulators_per_key is not implemented for MultiProcLocalBackend"
        )

    def reduce_per_key(self, col, combine_fn: Callable, stage_name: str):
        raise NotImplementedError(
            "reduce_per_key is not implemented for MultiProcLocalBackend")

    def flatten(self, col1, col2, stage_name: str = None):
        return itertools.chain(col1, col2)

    def distinct(self, col, stage_name: str):

        def generator():
            for v in set(col):
                yield v

        return generator()

    def to_list(self, col, stage_name: str):
        raise NotImplementedError(
            "to_list is not implemented for MultiProcLocalBackend")


class Annotator(abc.ABC):
    """Interface to annotate a PipelineDP pipeline.

    Call register_annotator() with your custom Annotator to annotate your
    pipeline."""

    @abc.abstractmethod
    def annotate(self, col, stage_name: str, **kwargs):
        """Annotates a collection.

        Args:
          stage_name: annotation stage_name, it needs to be correctly propagated
          kwargs: additional arguments about the current aggregation.

        Returns:
            The input collection after applying an annotation.
        """


_annotators = []


def register_annotator(annotator: Annotator):
    _annotators.append(annotator)
