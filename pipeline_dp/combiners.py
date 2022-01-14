import abc


class Combiner(abc.ABC):
    """Base class for all combiners.

    Combiners are objects that encapsulate aggregations and computations of
    differential private metrics. Combiners use accumulators to do aggregations.
    Combiners contain logic, while accumulators contain data.
    The API of combiners are inspired by Apache Beam CombineFn class.
    """

    @abc.abstractmethod
    def create_accumulator(self, values):
        """Creates accumulator from 'values'."""
        pass

    @abc.abstractmethod
    def merge_accumulators(self, accumulator1, accumulator2):
        """Merges the accumulators and returns accumulator."""
        pass

    @abc.abstractmethod
    def compute_metrics(self, accumulator: 'Accumulator'):
        """Computes and returns the result of aggregation."""
        pass
