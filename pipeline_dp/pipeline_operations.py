"""Adapters for working with pipeline frameworks."""

import random
import collections
import numpy as np

import abc
import apache_beam as beam
import apache_beam.transforms.combiners as combiners
import collections
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

    def keys(self, col, stage_name: str):
        return col | stage_name >> beam.Keys()

    def values(self, col, stage_name: str):
        return col | stage_name >> beam.Values()

    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        return col | stage_name >> combiners.Sample.FixedSizePerKey(n)

    def count_per_element(self, col, stage_name: str):
        return col | stage_name >> combiners.Count.PerElement()


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

    def keys(self, col, stage_name: str):
        pass

    def values(self, col, stage_name: typing.Optional[str] = None):
        return (v for k, v in col)

    def sample_fixed_per_key(self, col, n: int,
                             stage_name: typing.Optional[str] = None):
        def sample_fixed_per_key_generator():
            for item in self.group_by_key(col):
                key = item[0]
                values = item[1]
                if len(values) > n:
                    sampled_indices = np.random.choice(range(len(values)), n,
                                                       replace=False)
                    values = [values[i] for i in sampled_indices]
                yield key, values

        return sample_fixed_per_key_generator()

    def count_per_element(self, col, stage_name: typing.Optional[str] = None):
        yield from collections.Counter(col).items()
