import abc
import copy

import pipeline_dp
from pipeline_dp import dp_computations
from pipeline_dp import budget_accounting
import numpy as np
from collections import namedtuple


class Combiner(abc.ABC):
    """Base class for all combiners.

    Combiners are objects that encapsulate aggregations and computations of
    differential private metrics. Combiners use accumulators to store the
    aggregation state. Combiners contain logic, while accumulators contain data.
    The API of combiners are inspired by Apache Beam CombineFn class.
    https://beam.apache.org/documentation/transforms/python/aggregation/combineperkey/#example-5-combining-with-a-combinefn

    Here's how PipelineDP uses combiners to performs an aggregation on some
    dataset X:
    1. Split dataset X on sub-datasets which might be kept in memory.
    2. Call create_accumulators() for each sub-dataset and keep resulting accumulators.
    3. Choosing any pair of accumulators and merge them by calling merge_accumulators().
    4. Continue 3 until 1 accumulator is left.
    5. Call compute_metrics() for the accumulator that left.

    Assumption: merge_accumulators is associative binary operation.

    The type of accumulator depends on the aggregation performed by this Combiner.
    For example, this can be a primitive type (e.g. int for a simple "count"
    aggregation) or a more complex structure (e.g. for "mean")
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

    Wraps all the information needed by the combiner to do the
    differentially-private computation, e.g. privacy budget and mechanism.

    Note: 'aggregate_params' is copied.
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
        return count1 + count2

    def compute_metrics(self, count: int) -> float:
        return dp_computations.compute_dp_count(count,
                                                self._params.mean_var_params)


class MeanCombiner(Combiner):
    """Combiner for computing DP Mean. Also returns sum and count in addition to
    the mean.

    The type of the accumulator is a tuple(count: int, sum: float) that holds
    the count and sum of elements in the dataset for which this accumulator is
    computed.
    """

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values) -> (int, float):
        return len(values), np.clip(values, self._params.aggregate_params.low,
                                    self._params.aggregate_params.high).sum()

    def merge_accumulators(self, accum1: tuple, accum2: tuple):
        return accum1[0] + accum2[0], accum1[1] + accum2[1]

    def compute_metrics(self, accum: tuple) -> namedtuple:
        noisy_count, noisy_sum, noisy_mean = dp_computations.compute_dp_mean(
            accum[0], accum[1], self._params.mean_var_params)
        MeanTuple = namedtuple('MeanTuple', ['count', 'sum', 'mean'])
        return MeanTuple(count=noisy_count, sum=noisy_sum, mean=noisy_mean)
