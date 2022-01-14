import abc


class Combiner(abc.ABC):
    """Base class for all combiners.

  Combiners are objects that encapsulate aggregations and computations of
  differential private metrics. Combiners use accumulators to do aggregations.
  Combiners contain logic, while accumulators contain data.
  The API of combiners are inspired by Apache Beam CombineFn class.
  """

    @abc.abstractmethod
    def add_value(self, value):
        """Adds 'value' to accumulate.
    Args:
      value: value to be added.

    Returns: self.
    """
        pass

    def _check_mergeable(self, accumulator: 'Accumulator'):
        if not isinstance(accumulator, type(self)):
            raise TypeError(
                f"The accumulator to be added is not of the same type: "
                f"{accumulator.__class__.__name__} != "
                f"{self.__class__.__name__}")

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
