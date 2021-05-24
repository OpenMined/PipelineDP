"""Adapters for working with pipeline frameworks."""

import random

import abc
import apache_beam as beam
import apache_beam.transforms.combiners as combiners


class PipelineOperations(abc.ABC):
    """Interface for pipeline frameworks adapters."""

    @abc.abstractmethod
    def map(self, col, fn, stage_name: str):
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
    def filter_partitions(self, col, public_partitions, stage_name: str):
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

    def map_tuple(self, col, fn, stage_name: str):
        return col | stage_name >> beam.MapTuple(fn)

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

    def filter_partitions(self, col, public_partitions, data_extractors, stage_name: str):
        """Filter out the partitions that are not not meant to be public.

        Args:
          col: input collection
          public_partitions: collection of public partition keys
          stage_name: name of the stage

        Returns:
          A filtered collection containing only data belonging to public_partitions

        """
        class PartitionsFilterJoin(beam.DoFn):
          def process(self, joined_data):
            key, rest = joined_data
            values, is_public = rest.get('values'), rest.get('is_public')

            # TODO the Issue #4 says this is blocked on other tasks. Revisit
            # this once unblocked
            if not values:
              values = [None]

            if is_public:
              for value in values:
                yield key, value

        def is_public(col):
          return col[0] in public_partitions

        # no filtering if no public partitions are specified
        if not public_partitions:
          return col

        col = col | beam.Map(lambda x: (data_extractors.partition_extractor(x), x))

        pp_type = type(public_partitions)
        if pp_type is list or pp_type is set:
          return col | beam.Filter(is_public)
        else:
          public_partitions = public_partitions | beam.Map(lambda x: (x, True))
          return ({'values': col, 'is_public': public_partitions} | beam.CoGroupByKey() | beam.ParDo(PartitionsFilterJoin()))

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

    def filter_partitions(self, rdd, public_partitions, data_extractors, stage_name: str = None):
        pass

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

    def map(self, col, fn, stage_name: str = None):
        return map(fn, col)

    def map_tuple(self, col, fn, stage_name: str = None):
        return (fn(k, v) for k, v in col)

    def map_values(self, col, fn, stage_name: str):
        pass

    def group_by_key(self, col, stage_name: str):
        pass

    def filter(self, col, fn, stage_name: str):
        pass

    def filter_partitions(self, col, public_partitions, data_extractors, stage_name: str):
        pass

    def keys(self, col, stage_name: str):
        pass

    def values(self, col, stage_name: str):
        pass

    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        pass

    def count_per_element(self, col, stage_name: str):
        pass
