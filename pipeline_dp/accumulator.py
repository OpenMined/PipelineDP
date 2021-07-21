import abc
import typing
import pickle
from dataclasses import dataclass
from functools import reduce
from dataclasses import dataclass
import pipeline_dp


@dataclass
class AccumulatorParams:
    accumulator_type: type
    constructor_params: typing.Any


def merge(accumulators: typing.Iterable['Accumulator']) -> 'Accumulator':
    """Merges the accumulators."""
    return reduce(lambda acc1, acc2: acc1.add_accumulator(acc2), accumulators)


def create_accumulator_params(
    aggregation_params: pipeline_dp.AggregateParams,
    budget_accountant: pipeline_dp.BudgetAccountant
) -> typing.List[AccumulatorParams]:

    raise NotImplemented()  # implementation will be done later


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

       Sub-class implementation is responsible for checking that types of
       self and accumulator are the same.
      Args:
        accumulator:

      Returns: self
    """
        pass

    @abc.abstractmethod
    def compute_metrics(self):
        """Computes and returns the result of aggregation."""
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
    CompoundAccumulator delegates all operations to the internal accumulators.
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

    The expectation is that the internal accumulators are of the same type and
    are in the same order."""

        if len(accumulator.accumulators) != len(self.accumulators):
            raise ValueError(
                "Accumulators in the input are not of the same size." +
                f" Expected size = {len(self.accumulators)}" +
                f" received size = {len(accumulator.accumulators)}.")

        for pos, (base_accumulator_type, to_add_accumulator_type) in enumerate(
                zip(self.accumulators, accumulator.accumulators)):
            if type(base_accumulator_type) != type(to_add_accumulator_type):
                raise TypeError(
                    "The type of the accumulators don't match at "
                    f"index {pos}. {type(base_accumulator_type).__name__} "
                    f"!= {type(to_add_accumulator_type).__name__}.")

        for (base_accumulator,
             to_add_accumulator) in zip(self.accumulators,
                                        accumulator.accumulators):
            base_accumulator.add_accumulator(to_add_accumulator)
        return self

    def compute_metrics(self):
        """Computes and returns a list of metrics computed by internal
    accumulators."""
        return [
            accumulator.compute_metrics() for accumulator in self.accumulators
        ]


class AccumulatorFactory:
    """Factory for producing the appropriate Accumulator depending on the
    AggregateParams and BudgetAccountant."""

    def __init__(self, params: pipeline_dp.AggregateParams,
                 budget_accountant: pipeline_dp.BudgetAccountant):
        self._params = params
        self._budget_accountant = budget_accountant

    def initialize(self):
        self._accumulator_params = create_accumulator_params(
            self._params, self._budget_accountant)

    def create(self, values: typing.List) -> Accumulator:
        accumulators = []
        for accumulator_param in self._accumulator_params:
            accumulators.append(
                accumulator_param.accumulator_type(
                    accumulator_param.constructor_params, values))

        # No need to create CompoundAccumulator if there is only 1 accumulator.
        if len(accumulators) == 1:
            return accumulators[0]

        return CompoundAccumulator(accumulators)


@dataclass
class CountParams:
    pass


class CountAccumulator(Accumulator):

    def __init__(self, params: CountParams, values):
        self._count = len(values)

    def add_value(self, value):
        self._count += 1

    def add_accumulator(self,
                        accumulator: 'CountAccumulator') -> 'CountAccumulator':
        self._count += accumulator._count
        return self

    def compute_metrics(self) -> float:
        # TODO: add differential privacy
        return self._count
