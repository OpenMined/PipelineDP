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

    Let we have some dataset X to aggregate. The workflow of running an
    aggregation with combiners on X is the following:
    1.Split dataset X on sub-datasets which might be kept in memory.
    2.Call create_accumulators() for each sub-dataset and keep resulting accumulators.
    3.Choosing any pair of accumulators and merge them by calling merge_accumulators().
    4.Continue 3 until 1 accumulator is left.
    5.Call compute_metrics() for the accumulator that left.

    Assumption: merge_accumulators is associative binary operation.

    The type of the accumulator is specific for each concrete Combiner.
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
    def compute_metrics(self, accumulator):
        """Computes and returns the result of aggregation."""
        pass


class CombinerParams:
    """Parameters for a combiner.

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
    """Combiner for computing DP Count.

    The type of the accumulator is int, which represents count of the elements
    in the dataset for which this accumulator is computed.
    """

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values) -> int:
        return len(values)

    def merge_accumulators(self, count1: int, count2: int):
        return count1 + count1

    def compute_metrics(self, count: int) -> float:
        return dp_computations.compute_dp_count(count,
                                                self._params.mean_var_params)
