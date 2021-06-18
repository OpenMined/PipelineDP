import abc
import typing

class Accumulator(abc.ABC):
  @classmethod
  def merge(cls, accumulators: typing.Iterable[
    'Accumulator'] = None) -> 'Accumulator':
    return cls(accumulators)

  @abc.abstractmethod
  def add_value(self, v):
    pass

  @abc.abstractmethod
  def add_accumulator(self, accumulator: 'Accumulator') -> 'Accumulator':
    pass

  @abc.abstractmethod
  def compute_metrics(self):
    pass


class CompoundAccumulator(Accumulator):
  def __init__(self, accumulators: typing.Iterable['Accumulator'] = None):
    self.accumulators = []
    if accumulators:
      self.accumulators = [accumulator_expanded for accumulator in accumulators
    for accumulator_expanded in
                         (accumulator.accumulators if isinstance(accumulator,
                                                     CompoundAccumulator)
                          else [accumulator])]

  def add_value(self, v):
    for accumulator in self.accumulators:
      accumulator.add_value(v)

  def add_accumulator(self, accumulator: 'CompoundAccumulator') -> \
    'CompoundAccumulator':
    self.accumulators.extend(accumulator.accumulators)
    return self

  def compute_metrics(self):
    return [accumulator.compute_metrics() for accumulator in self.accumulators]
