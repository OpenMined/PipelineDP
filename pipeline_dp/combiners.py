import abc
import copy
from typing import Iterable, Sized, Tuple, List

import pipeline_dp
from pipeline_dp import dp_computations
from pipeline_dp import budget_accounting
import numpy as np
import collections


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

    @abc.abstractmethod
    def metrics_names(self) -> List[str]:
        """Return the list of names of the metrics this combiner computes"""


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

    def compute_metrics(self, count: AccumulatorType) -> dict:
        return {
            'count':
                dp_computations.compute_dp_count(count,
                                                 self._params.mean_var_params)
        }

    def metrics_names(self) -> List[str]:
        return ['count']


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

    def compute_metrics(self, accumulator: AccumulatorType) -> dict:
        return {
            'privacy_id_count':
                dp_computations.compute_dp_count(accumulator,
                                                 self._params.mean_var_params)
        }

    def metrics_names(self) -> List[str]:
        return ['privacy_id_count']


class SumCombiner(Combiner):
    """Combiner for computing dp sum.

    the type of the accumulator is int, which represents sum of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = float

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        return np.clip(values, self._params.aggregate_params.min_value,
                       self._params.aggregate_params.max_value).sum()

    def merge_accumulators(self, sum1: AccumulatorType, sum2: AccumulatorType):
        return sum1 + sum2

    def compute_metrics(self, sum: AccumulatorType) -> dict:
        return {
            'sum':
                dp_computations.compute_dp_sum(sum,
                                               self._params.mean_var_params)
        }

    def metrics_names(self) -> List[str]:
        return ['sum']


class MeanCombiner(Combiner):
    """Combiner for computing DP Mean. Also returns sum and count in addition to
    the mean.
    The type of the accumulator is a tuple(count: int, sum: float) that holds
    the count and sum of elements in the dataset for which this accumulator is
    computed.
    """
    AccumulatorType = Tuple[int, float]

    def __init__(self, params: CombinerParams,
                 metrics_to_compute: Iterable[str]):
        self._params = params
        if len(metrics_to_compute) != len(set(metrics_to_compute)):
            raise ValueError(f"{metrics_to_compute} cannot contain duplicates")
        for metric in metrics_to_compute:
            mean_metrics = ['count', 'sum', 'mean']
            if metric not in mean_metrics:
                raise ValueError(f"{metric} should be one of {mean_metrics}")
        if 'mean' not in metrics_to_compute:
            raise ValueError(
                f"one of the {metrics_to_compute} should be 'mean'")
        self._metrics_to_compute = metrics_to_compute

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        return len(values), np.clip(
            values, self._params.aggregate_params.min_value,
            self._params.aggregate_params.max_value).sum()

    def merge_accumulators(self, accum1: AccumulatorType,
                           accum2: AccumulatorType):
        count1, sum1 = accum1
        count2, sum2 = accum2
        return count1 + count2, sum1 + sum2

    def compute_metrics(self, accum: AccumulatorType) -> dict:
        total_count, total_sum = accum
        noisy_count, noisy_sum, noisy_mean = dp_computations.compute_dp_mean(
            total_count, total_sum, self._params.mean_var_params)
        mean_dict = {'mean': noisy_mean}
        if 'count' in self._metrics_to_compute:
            mean_dict['count'] = noisy_count
        if 'sum' in self._metrics_to_compute:
            mean_dict['sum'] = noisy_sum
        return mean_dict

    def metrics_names(self) -> List[str]:
        return self._metrics_to_compute


# Cache for namedtuple types. It is should be used only in
# '_get_or_create_named_tuple()' function.
_named_tuple_cache = {}


def _get_or_create_named_tuple(type_name: str,
                               field_names: tuple) -> 'MetricsTuple':
    """Creates namedtuple type with a custom serializer."""

    # The custom serializer is required for supporting serialization of
    # namedtuples in Apache Beam.
    cache_key = (type_name, field_names)
    named_tuple = _named_tuple_cache.get(cache_key)
    if named_tuple is None:
        named_tuple = collections.namedtuple(type_name, field_names)
        named_tuple.__reduce__ = lambda self: (_get_or_create_named_tuple,
                                               (type_name, field_names),
                                               tuple(self))
        _named_tuple_cache[cache_key] = named_tuple
    return named_tuple


class CompoundCombiner(Combiner):
    """Combiner for computing a set of dp aggregations.

    CompoundCombiner contains one or more combiners of other types for computing
    multiple metrics. For example it can contain [CountCombiner, SumCombiner].
    CompoundCombiner delegates all operations to the internal combiners.

    In case one the of combiners is MeanCombiner, which computes count and sum
    in addition to mean, output_count and output_sum should be set to True if
    they are to be outputted from MeanCombiner.

    The type of the accumulator is a a tuple of int and an iterable.
    The first int represents the privacy id count. The second iterable
    contains accumulators from internal combiners.

    Returns:
        namedtuple (MetricsTuple) whose fields are one or more of
        pipeline_dp.Metrics, depending on what the combiners compute. E.g., if
        CompoundCombiner has a PrivacyIdCountCombiner that computes
        'privacy_id_count' and a MeanCombiner that computes 'mean' and 'sum',
        CompoundCombiner will return a MetricsTuple(
            privacy_id_count=dp_privacy_id_count,
            sum=dp_sum,
            count=dp_mean,
        )
    """

    AccumulatorType = Tuple[int, Tuple]

    def __init__(self, combiners: Iterable['Combiner']):
        self._combiners = combiners
        self._metrics_to_compute = []
        for combiner in self._combiners:
            self._metrics_to_compute.extend(combiner.metrics_names())
        if len(self._metrics_to_compute) != len(set(self._metrics_to_compute)):
            raise ValueError(
                f"two combiners in {combiners} cannot compute the same metrics")
        self._metrics_to_compute = tuple(self._metrics_to_compute)
        self._MetricsTuple = _get_or_create_named_tuple(
            "MetricsTuple", self._metrics_to_compute)

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

    def compute_metrics(self, compound_accumulator: AccumulatorType):
        privacy_id_count, accumulator = compound_accumulator
        # Concatenates output of combiners, raises Exception if there are any
        # duplicates.
        combined_metrics = {}
        for combiner, acc in zip(self._combiners, accumulator):
            metrics_for_combiner = combiner.compute_metrics(acc)
            for metric, value in metrics_for_combiner.items():
                if metric in combined_metrics:
                    raise Exception(
                        f"{metric} computed by {combiner} was already computed by another combiner"
                    )
            combined_metrics.update(metrics_for_combiner)
        return self._MetricsTuple(**combined_metrics)

    def metrics_names(self) -> List[str]:
        return self._metrics_to_compute


def create_compound_combiner(
        aggregate_params: pipeline_dp.AggregateParams,
        budget_accountant: budget_accounting.BudgetAccountant
) -> CompoundCombiner:
    combiners = []
    mechanism_type = aggregate_params.noise_kind.convert_to_mechanism_type()

    if pipeline_dp.Metrics.MEAN in aggregate_params.metrics:
        budget_mean = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        metrics_to_compute = ['mean']
        if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
            metrics_to_compute.append('count')
        if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
            metrics_to_compute.append('sum')
        combiners.append(
            MeanCombiner(CombinerParams(budget_mean, aggregate_params),
                         metrics_to_compute))
    else:
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

    return CompoundCombiner(combiners)
