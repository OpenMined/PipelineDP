import abc
import copy
from typing import Iterable, Sized, Tuple

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

    @abc.abstractmethod
    def merge_accumulators(self, accumulator1, accumulator2):
        """Merges the accumulators and returns accumulator."""

    @abc.abstractmethod
    def compute_metrics(self, accumulator):
        """Computes and returns the result of aggregation."""


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
            self.eps, self.delta, self.aggregate_params.min_value,
            self.aggregate_params.max_value,
            self.aggregate_params.max_partitions_contributed,
            self.aggregate_params.max_contributions_per_partition,
            self.aggregate_params.noise_kind)


class CountCombiner(Combiner):
    """Combiner for computing DP Count.

    The type of the accumulator is int, which represents count of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = int

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values: Sized) -> AccumulatorType:
        return len(values)

    def merge_accumulators(self, count1: AccumulatorType,
                           count2: AccumulatorType):
        return count1 + count2

    def compute_metrics(self, count: AccumulatorType) -> float:
        return dp_computations.compute_dp_count(count,
                                                self._params.mean_var_params)


class PrivacyIdCountCombiner(Combiner):
    """Combiner for computing DP privacy id count.
    The type of the accumulator is int, which represents count of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = int

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values: Sized) -> AccumulatorType:
        return 1 if values else 0

    def merge_accumulators(self, accumulator1: AccumulatorType,
                           accumulator2: AccumulatorType):
        return accumulator1 + accumulator2

    def compute_metrics(self, accumulator: AccumulatorType) -> float:
        return dp_computations.compute_dp_count(accumulator,
                                                self._params.mean_var_params)


class SumCombiner(Combiner):
    """Combiner for computing dp sum.

    the type of the accumulator is int, which represents sum of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = float

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values: Iterable[float]) -> 'AccumulatorType':
        return np.clip(values, self._params.aggregate_params.min_value,
                       self._params.aggregate_params.max_value).sum()

    def merge_accumulators(self, sum1: AccumulatorType, sum2: AccumulatorType):
        return sum1 + sum2

    def compute_metrics(self, sum: AccumulatorType) -> float:
        return dp_computations.compute_dp_sum(sum, self._params.mean_var_params)


MeanTuple = namedtuple('MeanTuple', ['count', 'sum', 'mean'])


class MeanCombiner(Combiner):
    """Combiner for computing DP Mean. Also returns sum and count in addition to
    the mean.
    The type of the accumulator is a tuple(count: int, sum: float) that holds
    the count and sum of elements in the dataset for which this accumulator is
    computed.
    """
    AccumulatorType = Tuple[int, float]

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        return len(values), np.clip(
            values, self._params.aggregate_params.min_value,
            self._params.aggregate_params.max_value).sum()

    def merge_accumulators(self, accum1: AccumulatorType,
                           accum2: AccumulatorType):
        count1, sum1 = accum1
        count2, sum2 = accum2
        return count1 + count2, sum1 + sum2

    def compute_metrics(self, accum: AccumulatorType) -> namedtuple:
        total_count, total_sum = accum
        noisy_count, noisy_sum, noisy_mean = dp_computations.compute_dp_mean(
            total_count, total_sum, self._params.mean_var_params)
        return MeanTuple(count=noisy_count, sum=noisy_sum, mean=noisy_mean)


class CompoundCombiner(Combiner):
    """Combiner for computing a set of dp aggregations.

    CompoundCombiner contains one or more combiners of other types for computing multiple metrics.
    For example it can contain [CountCombiner, SumCombiner].
    CompoundCombiner delegates all operations to the internal combiners.

    The type of the accumulator is a a tuple of int and an iterable.
    The first int represents the privacy id count. The second iterable
    contains accumulators from internal combiners.
    """

    AccumulatorType = Tuple[int, Tuple]

    def __init__(self, combiners: Iterable['Combiner']):
        self._combiners = combiners

    def create_accumulator(self, values) -> AccumulatorType:
        return (1,
                tuple(
                    combiner.create_accumulator(values)
                    for combiner in self._combiners))

    def merge_accumulators(
            self, compound_accumulator1: AccumulatorType,
            compound_accumulator2: AccumulatorType) -> AccumulatorType:
        merged_accumulators = []
        privacy_id_count1, accumulator1 = compound_accumulator1
        privacy_id_count2, accumulator2 = compound_accumulator2
        for combiner, acc1, acc2 in zip(self._combiners, accumulator1,
                                        accumulator2):
            merged_accumulators.append(combiner.merge_accumulators(acc1, acc2))
        return (privacy_id_count1 + privacy_id_count2,
                tuple(merged_accumulators))

    def compute_metrics(
            self, compound_accumulator: AccumulatorType) -> Tuple[int, list]:
        privacy_id_count, accumulator = compound_accumulator
        metrics = []
        for combiner, acc in zip(self._combiners, accumulator):
            metrics.append(combiner.compute_metrics(acc))
        return (privacy_id_count, metrics)


def create_compound_combiner(
        aggregate_params: pipeline_dp.AggregateParams,
        budget_accountant: budget_accounting.BudgetAccountant
) -> CompoundCombiner:
    combiners = []
    mechanism_type = aggregate_params.noise_kind.convert_to_mechanism_type()

    if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
        budget_count = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        combiners.append(
            CountCombiner(CombinerParams(budget_count, aggregate_params)))
    if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
        budget_sum = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        combiners.append(
            SumCombiner(CombinerParams(budget_sum, aggregate_params)))
    if pipeline_dp.Metrics.PRIVACY_ID_COUNT in aggregate_params.metrics:
        budget_privacy_id_count = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        combiners.append(
            PrivacyIdCountCombiner(
                CombinerParams(budget_privacy_id_count, aggregate_params)))
    if pipeline_dp.Metrics.MEAN in aggregate_params.metrics:
        budget_mean = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        combiners.append(
            MeanCombiner(CombinerParams(budget_mean, aggregate_params)))
    return CompoundCombiner(combiners)
