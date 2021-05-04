"""Adapters for working with pipeline frameworks."""

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

class LocalPipelineOperations(PipelineOperations):
  """Local Pipeline adapter."""
  
  def map(self, col, fn, stage_name: str):
    pass

  def map_tuple(self, col, fn, stage_name: str):
    pass

  def map_values(self, col, fn, stage_name: str):
    pass

  def group_by_key(self, col, stage_name: str):
    pass

  def filter(self, col, fn, stage_name: str):
    pass

  def keys(self, col, stage_name: str):
    pass

  def values(self, col, stage_name: str):
    pass

  def sample_fixed_per_key(self, col, n: int, stage_name: str):
    pass

  def count_per_element(self, col, stage_name: str):
    pass