"""Adapters for working with pipeline frameworks."""

from enum import Enum
from functools import partial
import os
import multiprocessing as mp
from pipeline_dp import accumulator
import random
import numpy as np

import abc
import apache_beam as beam
import apache_beam.transforms.combiners as combiners
import collections
import pipeline_dp.accumulator as accumulator
import typing
from typing import Any, Optional, Callable, Tuple
import collections
import itertools


class PipelineOperations(abc.ABC):
    """Interface for pipeline frameworks adapters."""

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
        """Filters out nonpublic partitions.

        Args:
          col: collection with elements (partition_key, data).
          keys_to_keep: collection of public partition keys,
            both local (currently `list` and `set`) and distributed collections are supported
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
        pass

    @abc.abstractmethod
    def count_per_element(self, col, stage_name: str):
        pass

    @abc.abstractmethod
    def reduce_accumulators_per_key(self, col, stage_name: str):
        """Reduces the input collection so that all elements per each key are merged.

        Args:
          col: input collection which contains tuples (key, accumulator)
          stage_name: name of the stage

        Returns:
          A collection of tuples (key, accumulator).

        """
        pass


class BeamOperations(PipelineOperations):
    """Apache Beam adapter."""

    def map(self, col, fn, stage_name: str):
        return col | stage_name >> beam.Map(fn)

    def flat_map(self, col, fn, stage_name: str):
        return col | stage_name >> beam.FlatMap(fn)

    def map_tuple(self, col, fn, stage_name: str):
        return col | stage_name >> beam.Map(lambda x: fn(*x))

    def map_values(self, col, fn, stage_name: str):
        return col | stage_name >> beam.MapTuple(lambda k, v: (k, fn(v)))

    def group_by_key(self, col, stage_name: str):
        """Group the values for each key in the PCollection into a single sequence.

        Args:
          col: input collection
          stage_name: name of the stage

        Returns:
          An PCollection of tuples in which the type of the second item is list.

        """
        return col | stage_name >> beam.GroupByKey()

    def filter(self, col, fn, stage_name: str):
        return col | stage_name >> beam.Filter(fn)

    def filter_by_key(self, col, keys_to_keep, data_extractors,
                      stage_name: str):

        class PartitionsFilterJoin(beam.DoFn):

            def process(self, joined_data):
                key, rest = joined_data
                values, is_public = rest.get(VALUES), rest.get(IS_PUBLIC)

                # TODO the Issue #4 says this is blocked on other tasks. Revisit
                # this once unblocked
                if not values:
                    return

                if is_public:
                    for value in values:
                        yield key, value

        def has_public_partition_key(pk_val):
            return pk_val[0] in keys_to_keep

        # define constants for using as keys in CoGroupByKey
        VALUES, IS_PUBLIC = 0, 1

        if keys_to_keep is None:
            raise TypeError("Must provide a valid keys to keep")

        col = col | "Mapping data by partition" >> beam.Map(
            lambda x: (data_extractors.partition_extractor(x), x))

        if isinstance(keys_to_keep, (list, set)):
            # Keys to keep are in memory.
            if not isinstance(keys_to_keep, set):
                keys_to_keep = set(keys_to_keep)
            return col | "Filtering data from public partitions" >> beam.Filter(
                has_public_partition_key)

        # Public paritions are not in memory. Filter out with a join.
        keys_to_keep = (keys_to_keep | "Creating public_partitions PCollection"
                        >> beam.Map(lambda x: (x, True)))
        return ({
            VALUES: col,
            IS_PUBLIC: keys_to_keep
        } | "Aggregating elements by values and is_public partition flag " >>
                beam.CoGroupByKey() | "Filtering data from public partitions" >>
                beam.ParDo(PartitionsFilterJoin()))

    def keys(self, col, stage_name: str):
        return col | stage_name >> beam.Keys()

    def values(self, col, stage_name: str):
        return col | stage_name >> beam.Values()

    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        return col | stage_name >> combiners.Sample.FixedSizePerKey(n)

    def count_per_element(self, col, stage_name: str):
        return col | stage_name >> combiners.Count.PerElement()

    def reduce_accumulators_per_key(self, col, stage_name: str = None):
        # TODO: Use merge function from the accumulator framework.
        def merge_accumulators(accumulators):
            res = None
            for acc in accumulators:
                if res:
                    res.add_accumulator(acc)
                else:
                    res = acc
            return res

        return col | stage_name >> beam.CombinePerKey(merge_accumulators)


class SparkRDDOperations(PipelineOperations):
    """Apache Spark RDD adapter."""

    def map(self, rdd, fn, stage_name: str = None):
        return rdd.map(fn)

    def flat_map(self, rdd, fn, stage_name: str = None):
        return rdd.flatMap(fn)

    def map_tuple(self, rdd, fn, stage_name: str = None):
        return rdd.map(fn)

    def map_values(self, rdd, fn, stage_name: str = None):
        return rdd.mapValues(fn)

    def group_by_key(self, rdd, stage_name: str = None):
        """Group the values for each key in the RDD into a single sequence.

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

    def filter_by_key(self,
                      rdd,
                      keys_to_keep,
                      data_extractors,
                      stage_name: str = None):

        if keys_to_keep is None:
            raise TypeError("Must provide a valid keys to keep")

        rdd = rdd.map(lambda x: (data_extractors.partition_extractor(x), x))

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
        """Get fixed-size random samples for each unique key in an RDD of key-values.
        Sampling is not guaranteed to be uniform across partitions.

        Args:
          rdd: input RDD
          n: number of values to sample for each key
          stage_name: not used

        Returns:
          An RDD of tuples.

        """
        return rdd.mapValues(lambda x: [x]).reduceByKey(
            lambda x, y: random.sample(x + y, min(len(x) + len(y), n)))

    def count_per_element(self, rdd, stage_name: str = None):
        return rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: (x + y))

    def reduce_accumulators_per_key(self, rdd, stage_name: str = None):
        return rdd.reduceByKey(lambda acc1, acc2: acc1.add_accumulator(acc2))


class LocalPipelineOperations(PipelineOperations):
    """Local Pipeline adapter."""

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
        data_extractors,
        stage_name: typing.Optional[str] = None,
    ):
        return [(data_extractors.partition_extractor(x), x)
                for x in col
                if data_extractors.partition_extractor(x) in keys_to_keep]

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

    def reduce_accumulators_per_key(self, col, stage_name: str = None):
        return self.map_values(self.group_by_key(col), accumulator.merge)


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

        insert_row = partial(insert_row, (self.results_dict,))

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

        insert_row = partial(insert_row, (self.results_dict,))

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


class MultiProcLocalPipelineOperations(PipelineOperations):

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
                      data_extractors,
                      stage_name: typing.Optional[str] = None):

        def mapped_fn(captures, row):
            keys_to_keep_, data_extractors_ = captures
            key = data_extractors_.partition_extractor(row)
            return key, (key in keys_to_keep_)

        mapped_fn = partial(mapped_fn, (keys_to_keep, data_extractors))
        ordered_key_keep = self.map(col, mapped_fn, stage_name)
        return ((key, row)
                for row, (key, keep) in zip(col, ordered_key_keep)
                if keep)

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

        mapped_fn = partial(mapped_fn, (n,))
        groups = self.group_by_key(col, stage_name)
        return self.map(groups, mapped_fn, stage_name)

    def count_per_element(self, col, stage_name: typing.Optional[str] = None):
        return _LazyMultiProcCountIterator(col, self.chunksize, self.n_jobs,
                                           **self.pool_kwargs)

    def reduce_accumulators_per_key(self,
                                    col,
                                    stage_name: typing.Optional[str] = None):
        return self.map_values(col, accumulator.merge)
