import abc
import copy

import pipeline_dp
from pipeline_dp import dp_computations
from pipeline_dp import budget_accounting


class Combiner(abc.ABC):
    """Base class for all combiners.

    Combiners are objects that encapsulate aggregations and computations of
    differential private metrics. Combiners use accumulators to store the
    aggregation state. Combiners contain logic, while accumulators contain data.
    The API of combiners are inspired by Apache Beam CombineFn class.
    https://beam.apache.org/documentation/transforms/python/aggregation/combineperkey/#example-5-combining-with-a-combinefn
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


class CombinerParams:
    """Parameters for an combiner.

    Wraps epsilon and delta from the MechanismSpec which are lazily loaded.
    AggregateParams are copied into a MeanVarParams instance.
    """

    def __init__(self, spec: budget_accounting.MechanismSpec,
                 aggregate_params: pipeline_dp.AggregateParams):
        self._mechanism_spec = spec
        self.aggregate_params = copy.copy(aggregate_params)

    @property
    def eps(self):
        return self._mechanism_spec.eps

    @property
    def delta(self):
        return self._mechanism_spec.delta

    @property
    def mean_var_params(self):
        return dp_computations.MeanVarParams(
            self.eps, self.delta, self.aggregate_params.low,
            self.aggregate_params.high,
            self.aggregate_params.max_partitions_contributed,
            self.aggregate_params.max_contributions_per_partition,
            self.aggregate_params.noise_kind)


class CountCombiner(Combiner):

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values) -> int:
        return len(values)

    def merge_accumulators(self, accumulator1: int, accumulator2: int):
        return accumulator1 + accumulator2

    def compute_metrics(self, accumulator: int) -> float:
        return dp_computations.compute_dp_count(accumulator,
                                                self._params.mean_var_params)
