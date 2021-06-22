import abc
import typing


class Accumulator(abc.ABC):
  """
    Base class for all accumulators.
    Accumulators are objects that encapsulate aggregations and computations of
    differential private metrics.
  """

  @classmethod
  def merge(cls, accumulators: typing.Iterable[
    'Accumulator'] = None) -> 'Accumulator':
    """
    Merges the accumulators and creates an Accumulator.
    Args:
      accumulators:

    Returns: Accumulator instance with merged values.
    """
    return cls(accumulators)

  @abc.abstractmethod
  def add_value(self, value):
    """
    Adds the value to each of the accumulator.
    Args:
      value: value to be added.

    Returns: self.
    """
    pass

  @abc.abstractmethod
  def add_accumulator(self, accumulator: 'Accumulator') -> 'Accumulator':
    """
      Merges the accumulator
      The difference between this and the merge function is that here it
      accepts a single accumulator instead of a list removing the overhead of
      creating a list when merging.
      Args:
        accumulator:

      Returns: self
    """
    pass

  @abc.abstractmethod
  def compute_metrics(self):
    pass


class CompoundAccumulator(Accumulator):
  """
    CompoundAccumulator contains one or more accumulators of other types for
    computing multiple metrics. For example it can contain
    [CountAccumulator,  SumAccumulator].
    CompoundAccumulator delegates all operation to the internal accumulators.
  """

  def __init__(self, accumulators: typing.Iterable['Accumulator']):
    self.accumulators = []
    if accumulators:
      self.accumulators = accumulators

  def add_value(self,  value):
    # adds the value to each accumulator
    for accumulator in self.accumulators:
      accumulator.add_value(value)
    return self

  def add_accumulator(self, accumulator: 'CompoundAccumulator') -> \
          'CompoundAccumulator':
    # merges the accumulators of the CompoundAccumulators.
    # the expectation is that the input accumulators are of the same type and
    # are in the same order.

    if (len(accumulator.accumulators) != len(self.accumulators)
            or any([type( base_accumulator) != type(to_add_accumulator)
                    for (base_accumulator, to_add_accumulator)
                    in zip(self.accumulators, accumulator.accumulators)])):
      raise ValueError(
        "Accumulators in the input are not of the same size "
        + "or don't match the type/order of the base accumulators.")

    for (base_accumulator, to_add_accumulator) in zip(self.accumulators,
                                                      accumulator.accumulators):
      base_accumulator.add_accumulator(to_add_accumulator)
    return self

  def compute_metrics(self):
    # Computes the metrics for individual accumulator
    return [accumulator.compute_metrics() for accumulator in self.accumulators]
