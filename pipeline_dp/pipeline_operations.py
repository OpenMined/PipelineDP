"""Adapters for working with pipeline frameworks."""

import collections
import random
import numpy as np

import abc
import apache_beam as beam
import apache_beam.transforms.combiners as combiners
import collections
import pipeline_dp.accumulator as accumulator
import typing
import collections


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
        keys_to_keep = keys_to_keep | "Creating public_partitions PCollection" >> beam.Map(
            lambda x: (x, True))
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
                      keys_to_keep,
                      data_extractors,
                      stage_name: typing.Optional[str] = None):
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
