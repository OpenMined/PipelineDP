import abc
import typing


class Accumulator(abc.ABC):
  """
  Performs aggregations
  """

  @classmethod
  def merge(cls, accumulators: typing.Iterable[
    'Accumulator'] = None) -> 'Accumulator':
    """
    Merges the accumulators and creates an Accumulator
    Args:
      accumulators:

    Returns: Accumulator instance with merged values
    """
    return cls(accumulators)

  @abc.abstractmethod
  def add_value(self, v):
    """
    Adds the value to each of the accumulator constituting the
    CompoundAccumulator
    Args:
      v: value to be added

    Returns:self
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
  Performs aggregations for a compund accumulators
  """

  def __init__(self, accumulators: typing.Iterable['Accumulator'] = None):
    self.accumulators = []
    if accumulators:
      # flatten the accumulators if the input is a CompundAccumulator
      self.accumulators = [accumulator_expanded for accumulator in accumulators
                           for accumulator_expanded in
                           (accumulator.accumulators if isinstance(accumulator,
                                                                   CompoundAccumulator)
                            else [accumulator])]

  def add_value(self, v):
    # adds the value to each accumulator
    for accumulator in self.accumulators:
      accumulator.add_value(v)
    return self

  def add_accumulator(self, accumulator: 'CompoundAccumulator') -> \
    'CompoundAccumulator':
    # merges the accumulators of the CompoundAccumulators
    self.accumulators.extend(accumulator.accumulators)
    return self

  def compute_metrics(self):
    # Computes the metrics for individual accumulator
    return [accumulator.compute_metrics() for accumulator in self.accumulators]
