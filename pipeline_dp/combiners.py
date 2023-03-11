# Copyright 2022 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Combiners for computing DP aggregations."""

import abc
import copy
from typing import Callable, Iterable, Sized, Tuple, List, Union

import pipeline_dp
from pipeline_dp import dp_computations
from pipeline_dp import budget_accounting
import numpy as np
import collections
import pydp
from pydp.algorithms import quantile_tree

ArrayLike = Union[np.ndarray, List[float]]
ExplainComputationReport = Union[Callable, str, List[Union[Callable, str]]]


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

    Assumption: merge_accumulators is an associative binary operation.

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

    @abc.abstractmethod
    def explain_computation(self) -> ExplainComputationReport:
        pass


class CustomCombiner(Combiner, abc.ABC):
    """Base class for custom combiners.

    Warning: this is an experimental API. It might not work properly and it
    might be changed/removed without any notifications.

    Custom combiners are combiners provided by PipelineDP users and they allow
    to add custom DP aggregations for extending the PipelineDP functionality.

    The responsibility of CustomCombiner:
      1.Implement DP mechanism in `compute_metrics()`.
      2.If needed implement contribution bounding in
    `create_accumulator()`.

    Warning: this is an advanced feature that can break differential privacy
    guarantees if not implemented correctly.

    TODO(dvadym): after full implementation of Custom combiners to figure out
    whether CustomCombiner class is needed or Combiner can be used.
    """

    @abc.abstractmethod
    def request_budget(self,
                       budget_accountant: budget_accounting.BudgetAccountant):
        """Requests the budget.

        It is called by PipelineDP during the construction of the computations.
        The custom combiner can request a DP budget by calling
        'budget_accountant.request_budget()'. The budget object needs to be
        stored in a field of 'self'. It will be serialized and distributed
        to the workers together with 'self'.

        Warning: do not store 'budget_accountant' in 'self'. It is assumed to
        live in the driver process.
        """
        pass

    def set_aggregate_params(self,
                             aggregate_params: pipeline_dp.AggregateParams):
        """Sets aggregate parameters.

        The custom combiner can optionally use it for own DP parameter
        computations.
        """
        self._aggregate_params = aggregate_params

    def metrics_names(self) -> List[str]:
        """Metrics that 'self' computes.

        It returns the class name by default.
        """
        return self.__class__.__name__


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
    def scalar_noise_params(self):
        return dp_computations.ScalarNoiseParams(
            self.eps, self.delta, self.aggregate_params.min_value,
            self.aggregate_params.max_value,
            self.aggregate_params.min_sum_per_partition,
            self.aggregate_params.max_sum_per_partition,
            self.aggregate_params.max_partitions_contributed,
            self.aggregate_params.max_contributions_per_partition,
            self.aggregate_params.noise_kind)

    @property
    def additive_vector_noise_params(
            self) -> dp_computations.AdditiveVectorNoiseParams:
        return dp_computations.AdditiveVectorNoiseParams(
            eps_per_coordinate=self.eps / self.aggregate_params.vector_size,
            delta_per_coordinate=self.delta / self.aggregate_params.vector_size,
            max_norm=self.aggregate_params.vector_max_norm,
            l0_sensitivity=self.aggregate_params.max_partitions_contributed,
            linf_sensitivity=self.aggregate_params.
            max_contributions_per_partition,
            norm_kind=self.aggregate_params.vector_norm_kind,
            noise_kind=self.aggregate_params.noise_kind)


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
                dp_computations.compute_dp_count(
                    count, self._params.scalar_noise_params)
        }

    def metrics_names(self) -> List[str]:
        return ['count']

    def explain_computation(self) -> ExplainComputationReport:
        # TODO: add information in this and others combiners about amount of noise.
        return lambda: f"Computed count with (eps={self._params.eps} delta={self._params.delta})"


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
                dp_computations.compute_dp_count(
                    accumulator, self._params.scalar_noise_params)
        }

    def metrics_names(self) -> List[str]:
        return ['privacy_id_count']

    def explain_computation(self) -> ExplainComputationReport:
        return lambda: f"Computed privacy id count with (eps={self._params.eps} delta={self._params.delta})"


class SumCombiner(Combiner):
    """Combiner for computing dp sum.

    the type of the accumulator is int, which represents sum of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = float

    def __init__(self, params: CombinerParams):
        self._params = params
        self._bouding_per_partition = params.aggregate_params.bounds_per_partition_are_set

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        agg_params = self._params.aggregate_params
        if self._bouding_per_partition:
            # Sum values and clip.
            return np.clip(sum(values), agg_params.min_sum_per_partition,
                           agg_params.max_sum_per_partition)
        # Clip each value and sum.
        return np.clip(values, agg_params.min_value, agg_params.max_value).sum()

    def merge_accumulators(self, sum1: AccumulatorType, sum2: AccumulatorType):
        return sum1 + sum2

    def compute_metrics(self, sum_: AccumulatorType) -> dict:
        return {
            'sum':
                dp_computations.compute_dp_sum(sum_,
                                               self._params.scalar_noise_params)
        }

    def metrics_names(self) -> List[str]:
        return ['sum']

    def explain_computation(self) -> ExplainComputationReport:
        return lambda: f"Computed sum with (eps={self._params.eps} delta={self._params.delta})"


class MeanCombiner(Combiner):
    """Combiner for computing DP Mean.

    It can also return sum and count in addition to the mean.

    The type of the accumulator is a tuple(count: int, normalized_sum: float)
    that holds the count and normalized sum of elements in the dataset for which
    this accumulator is computed.
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
        min_value = self._params.aggregate_params.min_value
        max_value = self._params.aggregate_params.max_value
        middle = dp_computations.compute_middle(min_value, max_value)
        normalized_values = np.clip(values, min_value, max_value) - middle

        return len(values), normalized_values.sum()

    def merge_accumulators(self, accum1: AccumulatorType,
                           accum2: AccumulatorType):
        count1, sum1 = accum1
        count2, sum2 = accum2
        return count1 + count2, sum1 + sum2

    def compute_metrics(self, accum: AccumulatorType) -> dict:
        total_count, total_normalized_sum = accum
        noisy_count, noisy_sum, noisy_mean = dp_computations.compute_dp_mean(
            total_count, total_normalized_sum, self._params.scalar_noise_params)
        mean_dict = {'mean': noisy_mean}
        if 'count' in self._metrics_to_compute:
            mean_dict['count'] = noisy_count
        if 'sum' in self._metrics_to_compute:
            mean_dict['sum'] = noisy_sum
        return mean_dict

    def metrics_names(self) -> List[str]:
        return self._metrics_to_compute

    def explain_computation(self) -> ExplainComputationReport:
        return lambda: f"Computed mean with (eps={self._params.eps} delta={self._params.delta})"


class VarianceCombiner(Combiner):
    """Combiner for computing DP Variance.

    It can also return mean, sum and count in addition to the variance.

    The accumulator type is a
      tuple(count: int, normalized_sum: float, normalized_sum_of_squares: float)
    that holds the count, normalized sum and normalized sum of squares of
    elements in the dataset for which this accumulator is computed.
    """
    AccumulatorType = Tuple[int, float, float]

    def __init__(self, params: CombinerParams,
                 metrics_to_compute: Iterable[str]):
        self._params = params
        if len(metrics_to_compute) != len(set(metrics_to_compute)):
            raise ValueError(f"{metrics_to_compute} cannot contain duplicates")
        for metric in metrics_to_compute:
            variance_metrics = ['count', 'sum', 'mean', 'variance']
            if metric not in variance_metrics:
                raise ValueError(
                    f"{metric} should be one of {variance_metrics}")
        if 'variance' not in metrics_to_compute:
            raise ValueError(
                f"one of the {metrics_to_compute} should be 'variance'")
        self._metrics_to_compute = metrics_to_compute

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        min_value = self._params.aggregate_params.min_value
        max_value = self._params.aggregate_params.max_value
        middle = dp_computations.compute_middle(min_value, max_value)

        clipped_values = np.clip(values, min_value, max_value)
        normalized_values = clipped_values - middle

        return len(values), normalized_values.sum(), (
            normalized_values**2).sum()

    def merge_accumulators(self, accum1: AccumulatorType,
                           accum2: AccumulatorType):
        count1, sum1, sum_of_squares1 = accum1
        count2, sum2, sum_of_squares2 = accum2
        return count1 + count2, sum1 + sum2, sum_of_squares1 + sum_of_squares2

    def compute_metrics(self, accum: AccumulatorType) -> dict:
        total_count, total_normalized_sum, total_normalized_sum_of_squares = accum
        noisy_count, noisy_sum, noisy_mean, noisy_variance = dp_computations.compute_dp_var(
            total_count, total_normalized_sum, total_normalized_sum_of_squares,
            self._params.scalar_noise_params)
        variance_dict = {'variance': noisy_variance}
        if 'count' in self._metrics_to_compute:
            variance_dict['count'] = noisy_count
        if 'sum' in self._metrics_to_compute:
            variance_dict['sum'] = noisy_sum
        if 'mean' in self._metrics_to_compute:
            variance_dict['mean'] = noisy_mean
        return variance_dict

    def metrics_names(self) -> List[str]:
        return self._metrics_to_compute

    def explain_computation(self) -> ExplainComputationReport:
        return lambda: f"Computed variance with (eps={self._params.eps} delta={self._params.delta})"


class QuantileCombiner(Combiner):
    """Combiner for computing DP quantiles.

    It can compute any number of quantiles at once with the same accuracy.
    It uses QuantileTree algorithm from Google C++ DP library
    https://github.com/google/differential-privacy/blob/main/cc/algorithms/quantile-tree.h
    with the wrapper from PyDP.

    The accumulator is QuantileTree object serialized to string.
    """

    AccumulatorType = str

    def __init__(self, params, percentiles_to_compute: List[float]):
        self._params = params
        self._percentiles = percentiles_to_compute
        self._quantiles_to_compute = [p / 100 for p in percentiles_to_compute]

    def create_accumulator(self, values) -> AccumulatorType:
        tree = self._create_empty_quantile_tree()
        for value in values:
            tree.add_entry(value)
        return tree.serialize().to_bytes()

    def merge_accumulators(self, accumulator1: AccumulatorType,
                           accumulator2: AccumulatorType) -> AccumulatorType:
        tree = self._create_empty_quantile_tree()

        tree.merge(pydp._pydp.bytes_to_summary(accumulator1))
        tree.merge(pydp._pydp.bytes_to_summary(accumulator2))
        return tree.serialize().to_bytes()

    def compute_metrics(self, accumulator: AccumulatorType) -> AccumulatorType:
        tree = self._create_empty_quantile_tree()
        tree.merge(pydp._pydp.bytes_to_summary(accumulator))

        quantiles = tree.compute_quantiles(
            self._params.eps, self._params.delta,
            self._params.aggregate_params.max_partitions_contributed,
            self._params.aggregate_params.max_contributions_per_partition,
            self._quantiles_to_compute, self._noise_type())

        return dict([(name, value)
                     for name, value in zip(self.metrics_names(), quantiles)])

    def metrics_names(self) -> List[str]:

        def format_metric_name(p: float):
            int_p = int(round(p))
            if int_p == p:
                p = int_p
            else:
                p = str(p).replace('.', '_')
            return f"percentile_{p}"

        return list(map(format_metric_name, self._percentiles))

    def explain_computation(self) -> ExplainComputationReport:
        return lambda: f"Computed percentiles {self._percentiles} with (eps={self._params.eps} delta={self._params.delta})"

    def _create_empty_quantile_tree(self):
        # The default tree parameters taken from
        # https://github.com/google/differential-privacy/blob/605ec87bcbd4a536995b611132dbf4d341d2e91d/cc/algorithms/quantile-tree.h#L47
        DEFAULT_TREE_HEIGHT = 4
        DEFAULT_BRANCHING_FACTOR = 16
        return quantile_tree.QuantileTree(
            self._params.aggregate_params.min_value,
            self._params.aggregate_params.max_value, DEFAULT_TREE_HEIGHT,
            DEFAULT_BRANCHING_FACTOR)

    def _noise_type(self) -> str:
        noise_kind = self._params.aggregate_params.noise_kind
        if noise_kind == pipeline_dp.NoiseKind.LAPLACE:
            return "laplace"
        if noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
            return "gaussian"
        assert False, f"{noise_kind} is not support by PyDP quantile tree."


# Cache for namedtuple types. It should be used only in
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
        named_tuple.__reduce__ = lambda self: (_create_named_tuple_instance,
                                               (type_name, field_names,
                                                tuple(self)))
        _named_tuple_cache[cache_key] = named_tuple
    return named_tuple


def _create_named_tuple_instance(type_name: str, field_names: tuple, values):
    return _get_or_create_named_tuple(type_name, field_names)(*values)


class CompoundCombiner(Combiner):
    """Combiner for computing a set of dp aggregations.

    CompoundCombiner contains one or more combiners of other types for computing
    multiple metrics. For example, it can contain [CountCombiner, SumCombiner].
    CompoundCombiner delegates all operations to the internal combiners.

    In case one of the combiners is MeanCombiner, which computes count and sum
    in addition to mean, output_count and output_sum should be set to True if
    they are to be outputted from MeanCombiner. For VarianceCombiner you can
    additionally set output_mean to True.


    The type of the accumulator is a tuple of int and an iterable:
    - The first int represents the number of input rows. If rows are grouped by privacy ID,
      this will effectively be privacy ID count.
    - The second iterable contains accumulators from internal combiners.

    Returns:
        if return_named_tuple == False tuple with elements returned from the
            internal combiners in order in which internal combiners are specified.
        else returns namedtuple (MetricsTuple) whose fields are one or more of
            pipeline_dp.Metrics, depending on what the combiners compute.
            E.g., if CompoundCombiner has a PrivacyIdCountCombiner that computes
            'privacy_id_count' and a MeanCombiner that computes 'mean' and 'sum',
            CompoundCombiner will return a MetricsTuple(
                privacy_id_count=dp_privacy_id_count,
                sum=dp_sum,
                mean=dp_mean,
            )
    """

    AccumulatorType = Tuple[int, Tuple]

    def __init__(self, combiners: Iterable['Combiner'],
                 return_named_tuple: bool):
        self._combiners = list(combiners)
        self._metrics_to_compute = []
        self._return_named_tuple = return_named_tuple
        if not self._return_named_tuple:
            return
        # Creates namedtuple type based on what the internal combiners return.
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
        row_count1, accumulator1 = compound_accumulator1
        row_count2, accumulator2 = compound_accumulator2
        for combiner, acc1, acc2 in zip(self._combiners, accumulator1,
                                        accumulator2):
            merged_accumulators.append(combiner.merge_accumulators(acc1, acc2))
        return (row_count1 + row_count2, tuple(merged_accumulators))

    def compute_metrics(self, compound_accumulator: AccumulatorType):
        _, accumulator = compound_accumulator

        if not self._return_named_tuple:
            # returns tuple of what the internal combiners return.
            return tuple(
                combiner.compute_metrics(acc)
                for combiner, acc in zip(self._combiners, accumulator))

        # Concatenates output of the internal combiners into Namedtype, raises
        # Exception if there are any duplicates.
        combined_metrics = {}
        for combiner, acc in zip(self._combiners, accumulator):
            metrics_for_combiner = combiner.compute_metrics(acc)
            for metric, value in metrics_for_combiner.items():
                if metric in combined_metrics:
                    raise Exception(
                        f"{metric} computed by {combiner} was already computed by another combiner"
                    )
            combined_metrics.update(metrics_for_combiner)
        return _create_named_tuple_instance("MetricsTuple",
                                            tuple(combined_metrics.keys()),
                                            tuple(combined_metrics.values()))

    def metrics_names(self) -> List[str]:
        return self._metrics_to_compute

    def explain_computation(self) -> ExplainComputationReport:
        return [combiner.explain_computation() for combiner in self._combiners]


class VectorSumCombiner(Combiner):
    """Combiner for computing dp vector sum.

    The type of the accumulator is np.ndarray, which represents sum of the vectors of the same size
    for which this accumulator is computed.
    """
    AccumulatorType = np.ndarray

    def __init__(self, params: CombinerParams):
        self._params = params

    def create_accumulator(self,
                           values: Iterable[ArrayLike]) -> AccumulatorType:
        array_sum = None
        for val in values:
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            if val.shape != (self._params.aggregate_params.vector_size,):
                raise TypeError(
                    f"Shape mismatch: {val.shape} != {(self._params.aggregate_params.vector_size,)}"
                )
            if array_sum is None:
                array_sum = val
            else:
                array_sum += val
        return array_sum

    def merge_accumulators(self, array_sum1: AccumulatorType,
                           array_sum2: AccumulatorType):
        return array_sum1 + array_sum2

    def compute_metrics(self, array_sum: AccumulatorType) -> dict:
        return {
            'vector_sum':
                dp_computations.add_noise_vector(
                    array_sum, self._params.additive_vector_noise_params)
        }

    def metrics_names(self) -> List[str]:
        return ['vector_sum']

    def explain_computation(self) -> ExplainComputationReport:
        # TODO: add information about vector size, norm, amount of noise.
        lambda: f"Computed vector sum with (eps={self._params.eps} delta={self._params.delta})"


def create_compound_combiner(
        aggregate_params: pipeline_dp.AggregateParams,
        budget_accountant: budget_accounting.BudgetAccountant
) -> CompoundCombiner:
    combiners = []
    mechanism_type = aggregate_params.noise_kind.convert_to_mechanism_type()

    if pipeline_dp.Metrics.VARIANCE in aggregate_params.metrics:
        budget_variance = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        metrics_to_compute = ['variance']
        if pipeline_dp.Metrics.MEAN in aggregate_params.metrics:
            metrics_to_compute.append('mean')
        if pipeline_dp.Metrics.COUNT in aggregate_params.metrics:
            metrics_to_compute.append('count')
        if pipeline_dp.Metrics.SUM in aggregate_params.metrics:
            metrics_to_compute.append('sum')
        combiners.append(
            VarianceCombiner(CombinerParams(budget_variance, aggregate_params),
                             metrics_to_compute))
    elif pipeline_dp.Metrics.MEAN in aggregate_params.metrics:
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
    if pipeline_dp.Metrics.VECTOR_SUM in aggregate_params.metrics:
        budget_vector_sum = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        combiners.append(
            VectorSumCombiner(
                CombinerParams(budget_vector_sum, aggregate_params)))

    percentiles_to_compute = [
        metric.parameter
        for metric in aggregate_params.metrics
        if metric.is_percentile
    ]
    if percentiles_to_compute:
        budget_percentile = budget_accountant.request_budget(
            mechanism_type, weight=aggregate_params.budget_weight)
        combiners.append(
            QuantileCombiner(
                CombinerParams(budget_percentile, aggregate_params),
                percentiles_to_compute))

    return CompoundCombiner(combiners, return_named_tuple=True)


def create_compound_combiner_with_custom_combiners(
        aggregate_params: pipeline_dp.AggregateParams,
        budget_accountant: budget_accounting.BudgetAccountant,
        custom_combiners: Iterable[CustomCombiner]) -> CompoundCombiner:
    for combiner in custom_combiners:
        combiner.request_budget(budget_accountant)
        combiner.set_aggregate_params(aggregate_params)

    return CompoundCombiner(custom_combiners, return_named_tuple=False)
