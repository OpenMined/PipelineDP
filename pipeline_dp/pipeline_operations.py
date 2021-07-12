"""Adapters for working with pipeline frameworks."""

from functools import partial
import os
import multiprocessing as mp
from . import accumulator
import random
import numpy as np

import abc
import apache_beam as beam
import apache_beam.transforms.combiners as combiners
import typing


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
    def filter_by_key(self, col, public_partitions, stage_name: str):
        """Filters out nonpublic partitions.

        Args:
          col: collection with elements (partition_key, data).
          public_partitions: collection of public partition keys.
          stage_name: name of the stage.

        Returns:
          A filtered collection containing only data belonging to public_partitions.

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

    def filter_by_key(self, col, public_partitions, data_extractors,
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
            return pk_val[0] in public_partitions

        # define constants for using as keys in CoGroupByKey
        VALUES, IS_PUBLIC = 0, 1

        if public_partitions is None:
            raise TypeError("Must provide a valid public_partitions")

        col = col | "Mapping data by partition" >> beam.Map(
            lambda x: (data_extractors.partition_extractor(x), x))

        if isinstance(public_partitions, (list, set)):
            # Public partitions are in memory.
            if not isinstance(public_partitions, set):
                public_partitions = set(public_partitions)
            return col | "Filtering data from public partitions" >> beam.Filter(
                has_public_partition_key)

        # Public paritions are not in memory. Filter out with a join.
        public_partitions = public_partitions | "Creating public_partitions PCollection" >> beam.Map(
            lambda x: (x, True))
        return ({
            VALUES: col,
            IS_PUBLIC: public_partitions
        } | "Aggregating elements by values and is_public partition flag " >>
                beam.CoGroupByKey() | "Filtering data from public partitions"
                >> beam.ParDo(PartitionsFilterJoin()))

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
                      public_partitions,
                      data_extractors,
                      stage_name: str = None):
        NotImplementedError(
            "filter_by_key is not implemented in SparkRDDOperations")

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
        return rdd.mapValues(lambda x: [x])\
            .reduceByKey(lambda x, y: random.sample(x+y, min(len(x)+len(y), n)))

    def count_per_element(self, rdd, stage_name: str = None):
        return rdd.map(lambda x: (x, 1))\
            .reduceByKey(lambda x, y: (x + y))

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

    def filter_by_key(self,
                      col,
                      public_partitions,
                      data_extractors,
                      stage_name: typing.Optional[str] = None):
        return [(data_extractors.partition_extractor(x), x)
                for x in col
                if data_extractors.partition_extractor(x) in public_partitions]

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
        raise NotImplementedError()

class MultiProcLocalPipelineOperations(PipelineOperations):
    def __init__(self, n_jobs: typing.Optional[int]=None,
                chunksize: int=1,
                ordered=True, 
                **pool_kwargs):
        if n_jobs is None:
            n_jobs = len(os.sched_getaffinity())
        self.n_jobs = n_jobs
        self.chunksize = chunksize
        self.ordered = ordered
        self.pool = mp.Pool(n_jobs, **pool_kwargs)
        if ordered:
            self._pool_map_fn = self.pool.imap
        else:
            self._pool_map_fn = self.pool.imap_unordered

    def map(self, col, fn, stage_name: str):
        return self._pool_map_fn(fn, col, chunksize=self.chunksize)

    def flat_map(self, col, fn, stage_name: str):
        return (e for x in self.map(col, fn, stage_name) for e in x)

    def map_tuple(self, col, fn, stage_name: str):
        def mapped_fn(captures: typing.Tuple[typing.Callable], row):
            func, = captures
            return func(*row)
        mapped_fn = partial(mapped_fn, (fn, ))
        return self.map(col, mapped_fn, stage_name)

    def map_values(self, col, fn, stage_name: str):
        def mapped_fn(captures: typing.Tuple[typing.Callable], row):
            func, = captures
            return row[0], func(row[1])
        mapped_fn = partial(mapped_fn, (fn, ))
        return self.map(col, mapped_fn, stage_name)

    def group_by_key(self, col, stage_name: str):
        # NOTE - this cannot be implemented in an ordered manner without (almost) serial execution!
        #   both keys and groups will be out of order
        with mp.Manager() as manager:
            results_dict = manager.dict()
            def insert_row(captures, row):
                manager_, results_dict_ = captures
                key, val = row
                if results_dict_.get(key, None) is None:
                    results_dict_[key] = manager_.list()
                results_dict_[key].append(val)
            insert_row = partial(insert_row, (manager, results_dict))
            self.pool.map_async(insert_row, col, self.chunksize).wait()
            return ((k, v) for k, v in results_dict.items())

    def _filter_ordered(self, col, fn, stage_name: str):
        def mapped_fn(captures: typing.Tuple[typing.Callable], row):
            func, = captures
            return row, func(row)
        mapped_fn = partial(mapped_fn, (fn, ))
        return (
            row for row, keep in self.map(col, mapped_fn, stage_name) if keep
        )

    def _filter_unordered(self, col, fn, stage_name: str):
        # TODO - implement using multiprocessing.Queue
        #   details: We want to make the filtering happen on the workers themselves
        #       rather than on the main thread.
        return self._filter_ordered(col, fn, stage_name)

    def filter(self, col, fn, stage_name: str):
        if self.ordered:
            return self._filter_ordered(col, fn, stage_name)
        return self._filter_unordered(col, fn, stage_name)

    def _filter_by_key_ordered(self,
                                col,
                                public_partitions,
                                data_extractors,
                                stage_name: typing.Optional[str] = None):
        def mapped_fn(captures, row):
            public_partitions_, data_extractors_ = captures
            partition = data_extractors_.partition_extractor(row)
            return row, (partition in public_partitions_)
        mapped_fn = partial(mapped_fn, (public_partitions, data_extractors))

        return (
            row for row, keep in self.map(col, mapped_fn, stage_name) if keep
        )

    def _filter_by_key_unordered(self,
                                col,
                                public_partitions,
                                data_extractors,
                                stage_name: typing.Optional[str] = None):
        # TODO - implement using multiprocessing.Queue
        #   details: We want to make the filtering happen on the workers themselves
        #       rather than on the main thread.
        return self._filter_by_key_ordered(col, public_partitions, data_extractors, stage_name)

    def filter_by_key(self,
                        col,
                        public_partitions,
                        data_extractors,
                        stage_name: typing.Optional[str] = None):
        if self.ordered:
            return self._filter_by_key_ordered(col, public_partitions,
             data_extractors, stage_name)
        return self._filter_by_key_unordered(col, public_partitions, 
            data_extractors, stage_name)

    def keys(self, col, stage_name: str):
        # no point in passing through multiproc.
        return (k for k, v in col)

    def values(self, col, stage_name: str):
        # no point in passing through multiproc.
        return (v for k, v in col)

    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        def mapped_fn(captures, row):
            n_, = captures
            partition_key, values = row
            samples = values
            if len(samples) > n_:
                samples = random.sample(samples, n_)
            return partition_key, samples
        mapped_fn = partial(mapped_fn, (n,))
        groups = self.group_by_key(col, stage_name)
        return self.map(groups, mapped_fn, stage_name)

    def count_per_element(self, col, stage_name: str):
        groups = self.group_by_key(col, stage_name)
        return self.map_values(groups, len, stage_name)

    def reduce_accumulators_per_key(self, col, stage_name: str):
        """Reduces the input collection so that all elements per each key are merged.

            Args:
              col: input collection which contains tuples (key, accumulator)
              stage_name: name of the stage

            Returns:
              A collection of tuples (key, accumulator).

            """
        return self.map_values(col, accumulator.merge)
