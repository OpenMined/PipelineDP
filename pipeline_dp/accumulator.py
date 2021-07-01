import abc
import typing
import pickle
from functools import reduce


def merge(accumulators: typing.Iterable[
  'Accumulator'] = None) -> 'Accumulator':
  """Merges the accumulators.

  Args:
    accumulators:

  Returns: Accumulator instance with merged values.
  """
  unique_accumulator_types = {type(accumulator).__name__ for accumulator in
                              accumulators}
  if len(unique_accumulator_types) > 1:
    raise TypeError(
      "Accumulators should all be of the same type. Found accumulators of "
      + f"different types: ({','.join(unique_accumulator_types)}).")

  return reduce(lambda acc1, acc2: acc1.add_accumulator(acc2), accumulators)


class Accumulator(abc.ABC):
  """Base class for all accumulators.

    Accumulators are objects that encapsulate aggregations and computations of
    differential private metrics.
  """

  @abc.abstractmethod
  def add_value(self, value):
    """Adds the value to each of the accumulator.
    Args:
      value: value to be added.

    Returns: self.
    """
    pass

  @abc.abstractmethod
  def add_accumulator(self, accumulator: 'Accumulator') -> 'Accumulator':
    """Merges the accumulator to self and returns self.
      Args:
        accumulator:

      Returns: self
    """
    pass

  @abc.abstractmethod
  def compute_metrics(self):
    pass

  def serialize(self):
    return pickle.dumps(self)

  @classmethod
  def deserialize(cls, serialized_obj: str):
    deserialized_obj = pickle.loads(serialized_obj)
    if not isinstance(deserialized_obj, cls):
      raise TypeError("The deserialized object is not of the right type.")
    return deserialized_obj


class CompoundAccumulator(Accumulator):
  """Accumulator for computing multiple metrics.

    CompoundAccumulator contains one or more accumulators of other types for
    computing multiple metrics.
    For example it can contain [CountAccumulator,  SumAccumulator].
    CompoundAccumulator delegates all operation to the internal accumulators.
  """

  def __init__(self, accumulators: typing.Iterable['Accumulator']):
    self.accumulators = accumulators

  def add_value(self, value):
    for accumulator in self.accumulators:
      accumulator.add_value(value)
    return self

  def add_accumulator(self, accumulator: 'CompoundAccumulator') -> \
    'CompoundAccumulator':
    """Merges the accumulators of the CompoundAccumulators.
    The expectation is that the input accumulators are of the same type and
    are in the same order."""

    if len(accumulator.accumulators) != len(self.accumulators):
      raise ValueError(
        "Accumulators in the input are not of the same size."
        + f" Expected size = {len(self.accumulators)}"
        + f" received size = {len(accumulator.accumulators)}.")

    expected_type_order = ",".join([type(accumulator).__name__ for
                                    accumulator in self.accumulators])
    received_type_order = ",".join([type(accumulator).__name__ for
                                    accumulator in accumulator.accumulators])
    if any([base_accumulator_type != to_add_accumulator_type for
            base_accumulator_type, to_add_accumulator_type in
            zip(expected_type_order, received_type_order)]):
      raise TypeError(
        f"""Accumulators in the input don't match the type/order of the base accumulators. 
        Expected {expected_type_order}
        received {received_type_order}""")

    for (base_accumulator, to_add_accumulator) in zip(self.accumulators,
                                                      accumulator.accumulators):
      base_accumulator.add_accumulator(to_add_accumulator)
    return self

  def compute_metrics(self):
    """Computes and returns a list of metrics computed by internal
    accumulators."""
    return [accumulator.compute_metrics() for accumulator in self.accumulators]
