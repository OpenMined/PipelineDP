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
"""Utility Analysis per-partition Combiners."""

import abc
import copy
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union
import numpy as np
import math

import pipeline_dp
from pipeline_dp import budget_accounting
from pipeline_dp import dp_computations
from analysis import metrics
from analysis import poisson_binomial
from pipeline_dp import partition_selection

MAX_PROBABILITIES_IN_ACCUMULATOR = 100

# It corresponds to the aggregating per (privacy_id, partition_key).
# (count, sum, num_partition_privacy_id_contributes).
PreaggregatedData = Tuple[int, Union[float, tuple[float]], int]


class UtilityAnalysisCombiner(pipeline_dp.Combiner):

    @abc.abstractmethod
    def create_accumulator(self, data: Tuple[int, float, int]):
        """Creates an accumulator for data.

        Args:
            data is a Tuple containing:
              1) the count of the user's contributions for a single partition
              2) the sum of the user's contributions for the same partition
              3) the total number of partitions a user contributed to.

            Only COUNT, PRIVACY_ID_COUNT, SUM metrics can be supported with the
            current format of data.

        Returns:
            A tuple which is an accumulator.
        """

    def merge_accumulators(self, acc1: Tuple, acc2: Tuple):
        """Merges two tuples together additively."""
        return tuple(a + b for a, b in zip(acc1, acc2))

    def explain_computation(self):
        """No-op."""

    def metrics_names(self) -> List[str]:
        """Not used for utility analysis combiners."""
        return []


@dataclass
class SumOfRandomVariablesMoments:
    """Stores moments of sum of random independent random variables."""
    count: int
    expectation: float
    variance: float
    third_central_moment: float

    def __add__(
            self, other: 'SumOfRandomVariablesMoments'
    ) -> 'SumOfRandomVariablesMoments':
        return SumOfRandomVariablesMoments(
            self.count + other.count, self.expectation + other.expectation,
            self.variance + other.variance,
            self.third_central_moment + other.third_central_moment)


def _probabilities_to_moments(
        probabilities: List[float]) -> SumOfRandomVariablesMoments:
    """Computes moments of sum of independent bernoulli random variables."""
    exp = sum(probabilities)
    var = sum([p * (1 - p) for p in probabilities])
    third_central_moment = sum(
        [p * (1 - p) * (1 - 2 * p) for p in probabilities])

    return SumOfRandomVariablesMoments(len(probabilities), exp, var,
                                       third_central_moment)


@dataclass
class PartitionSelectionCalculator:
    """Computes probability of keeping the partition.

    Args:
        probabilities: probabilities that each specific user contributes to the
        partition after contribution bounding.
        moments: contains moments of the sum of independent random
        variables, which represent whether user contributes to the partition.

    Those variables are set mutually exclusive. If len(probabilities) <=
    MAX_PROBABILITIES_IN_ACCUMULATOR then 'probabilities' are used otherwise
    'moments'. The idea is that when the number of the contributions are small
    the sum of the random variables is far from Normal distribution and exact
    computations are performed, otherwise a Normal approximation based on
    moments is used.
    """
    probabilities: Optional[List[float]] = None
    moments: Optional[SumOfRandomVariablesMoments] = None

    def __post_init__(self):
        assert (self.probabilities is None) != (
            self.moments
            is None), "Only one of probabilities and moments must be set."

    def compute_probability_to_keep(self,
                                    partition_selection_strategy: pipeline_dp.
                                    PartitionSelectionStrategy, eps: float,
                                    delta: float,
                                    max_partitions_contributed: int,
                                    pre_threshold: Optional[int]) -> float:
        """Computes the probability that this partition is kept.

        If self.probabilities is set, then the computed probability is exact,
        otherwise it is an approximation computed from self.moments.
        """
        pmf = self._compute_pmf()
        ps_strategy = partition_selection.create_partition_selection_strategy(
            partition_selection_strategy, eps, delta,
            max_partitions_contributed, pre_threshold)
        probability = 0
        for i, prob in enumerate(pmf.probabilities, pmf.start):
            probability += prob * ps_strategy.probability_of_keep(i)
        return probability

    def _compute_pmf(self) -> poisson_binomial.PMF:
        """Computes the pmf of privacy id count in this partition after contribution bounding."""
        if self.probabilities:
            return poisson_binomial.compute_pmf(self.probabilities)

        moments = self.moments
        std = math.sqrt(moments.variance)
        skewness = 0 if std == 0 else moments.third_central_moment / std**3
        return poisson_binomial.compute_pmf_approximation(
            moments.expectation, std, skewness, moments.count)


# PartitionSelectionAccumulator = (probabilities, moments). These two elements
# exclusive:
# If len(probabilities) <= MAX_PROBABILITIES_IN_ACCUMULATOR then 'probabilities'
# are used otherwise 'moments'. For more details see docstring to
# PartitionSelectionCalculator.
PartitionSelectionAccumulator = Tuple[Optional[Tuple[float]],
                                      Optional[SumOfRandomVariablesMoments]]


def _merge_list(a: List, b: List) -> List:
    """Combines 2 lists in 1 and returns it.

    Warning: it modifies arguments.
    """
    # Extend the larger list for performance reasons.
    if len(a) >= len(b):
        a.extend(b)
        return a
    b.extend(a)
    return b


def _merge_partition_selection_accumulators(
        acc1: PartitionSelectionAccumulator,
        acc2: PartitionSelectionAccumulator) -> PartitionSelectionAccumulator:
    probs1, moments1 = acc1
    probs2, moments2 = acc2
    if ((probs1 is not None) and (probs2 is not None) and
            len(probs1) + len(probs2) <= MAX_PROBABILITIES_IN_ACCUMULATOR):
        return (_merge_list(probs1, probs2), None)

    if moments1 is None:
        moments1 = _probabilities_to_moments(probs1)
    if moments2 is None:
        moments2 = _probabilities_to_moments(probs2)

    return (None, moments1 + moments2)


class PartitionSelectionCombiner(UtilityAnalysisCombiner):
    """A combiner for utility analysis counts."""

    def __init__(self, spec: budget_accounting.MechanismSpec,
                 params: pipeline_dp.AggregateParams):
        self._spec = spec
        self._params = params

    def create_accumulator(self, sparse_acc: Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray]):
        count, sum_, n_partitions = sparse_acc
        max_partitions = self._params.max_partitions_contributed
        prob_keep_partition = np.where(
            n_partitions > 0, np.minimum(1, max_partitions / n_partitions), 0)
        acc = (list(prob_keep_partition), None)
        empty_acc = ([], None)
        # 'acc' can contain many probabilities and in that case it is better to
        # convert it to moments. The next line achieves this.
        return _merge_partition_selection_accumulators(acc, empty_acc)

    def merge_accumulators(
            self, acc1: PartitionSelectionAccumulator,
            acc2: PartitionSelectionAccumulator
    ) -> PartitionSelectionAccumulator:
        return _merge_partition_selection_accumulators(acc1, acc2)

    def compute_metrics(self, acc: PartitionSelectionAccumulator) -> float:
        """Computes the probability that the partition kept."""
        probs, moments = acc
        params = self._params
        spec = self._spec
        calculator = PartitionSelectionCalculator(probs, moments)
        return calculator.compute_probability_to_keep(
            params.partition_selection_strategy, spec.eps, spec.delta,
            params.max_partitions_contributed, params.pre_threshold)


class SumCombiner(UtilityAnalysisCombiner):
    """A combiner for utility analysis sums."""
    # (partition_sum, clipping_to_min_error, clipping_to_max_error,
    # expected_l0_bounding_error, var_cross_partition_error)
    AccumulatorType = Tuple[float, float, float, float, float]

    def __init__(self,
                 spec: budget_accounting.MechanismSpec,
                 params: pipeline_dp.AggregateParams,
                 metric: pipeline_dp.Metrics = pipeline_dp.Metrics.SUM,
                 i_column: Optional[int] = None):
        self._spec = spec
        self._params = copy.deepcopy(params)
        self._metric = metric
        self._i_column = i_column

    def create_accumulator(
            self, data: Tuple[np.ndarray, np.ndarray,
                              np.ndarray]) -> AccumulatorType:
        count, partition_sum, n_partitions = data
        if self._i_column != -1:
            # extract corresponding column in case of multi-column case.
            partition_sum = partition_sum[:, self._i_column]
        del count  # not used for SumCombiner
        min_bound = self._params.min_sum_per_partition
        max_bound = self._params.max_sum_per_partition
        max_partitions = self._params.max_partitions_contributed
        l0_prob_keep_contribution = np.where(
            n_partitions > 0, np.minimum(1, max_partitions / n_partitions), 0)
        per_partition_contribution = np.clip(partition_sum, min_bound,
                                             max_bound)
        per_partition_error = per_partition_contribution - partition_sum
        clipping_to_min_error = np.where(partition_sum < min_bound,
                                         per_partition_error, 0)
        clipping_to_max_error = np.where(partition_sum > max_bound,
                                         per_partition_error, 0)
        expected_l0_bounding_error = -per_partition_contribution * (
            1 - l0_prob_keep_contribution)
        var_cross_partition_error = per_partition_contribution**2 * l0_prob_keep_contribution * (
            1 - l0_prob_keep_contribution)

        return (partition_sum.sum().item(), clipping_to_min_error.sum().item(),
                clipping_to_max_error.sum().item(),
                expected_l0_bounding_error.sum().item(),
                var_cross_partition_error.sum().item())

    def compute_metrics(self, acc: AccumulatorType) -> metrics.SumMetrics:
        """Computes metrics based on the accumulator properties."""
        partition_sum, clipping_to_min_error, clipping_to_max_error, expected_l0_bounding_error, var_cross_partition_error = acc
        return metrics.SumMetrics(
            aggregation=self._metric,
            sum=partition_sum,
            clipping_to_min_error=clipping_to_min_error,
            clipping_to_max_error=clipping_to_max_error,
            expected_l0_bounding_error=expected_l0_bounding_error,
            std_l0_bounding_error=math.sqrt(var_cross_partition_error),
            std_noise=self._get_std_noise(),
            noise_kind=self._params.noise_kind)

    def get_sensitivities(self) -> dp_computations.Sensitivities:
        return dp_computations.compute_sensitivities_for_sum(self._params)

    def _get_std_noise(self) -> float:
        sensitivities = self.get_sensitivities()
        mechanism = dp_computations.create_additive_mechanism(
            self._spec, sensitivities)
        return mechanism.std


class CountCombiner(SumCombiner):
    """A combiner for utility analysis counts."""
    # (partition_sum, clipping_to_min_error, clipping_to_max_error,
    # expected_l0_bounding_error, var_cross_partition_error)
    AccumulatorType = Tuple[float, float, float, float, float]

    def __init__(self, mechanism_spec: budget_accounting.MechanismSpec,
                 params: pipeline_dp.AggregateParams):
        super().__init__(mechanism_spec, params, pipeline_dp.Metrics.COUNT)

    def create_accumulator(
        self, sparse_acc: Tuple[np.ndarray, np.ndarray,
                                np.ndarray]) -> AccumulatorType:
        count, _sum, n_partitions = sparse_acc
        data = None, count, n_partitions
        self._params.min_sum_per_partition = 0.0
        self._params.max_sum_per_partition = self._params.max_contributions_per_partition
        return super().create_accumulator(data)

    def get_sensitivities(self) -> dp_computations.Sensitivities:
        return dp_computations.compute_sensitivities_for_count(self._params)


class PrivacyIdCountCombiner(SumCombiner):
    """A combiner for utility analysis privacy ID counts."""
    # (partition_sum, clipping_to_min_error, clipping_to_max_error,
    # expected_l0_bounding_error, var_cross_partition_error)
    AccumulatorType = Tuple[float, float, float, float, float]

    def __init__(self, mechanism_spec: budget_accounting.MechanismSpec,
                 params: pipeline_dp.AggregateParams):
        super().__init__(mechanism_spec, params,
                         pipeline_dp.Metrics.PRIVACY_ID_COUNT)

    def create_accumulator(
        self, sparse_acc: Tuple[np.ndarray, np.ndarray,
                                np.ndarray]) -> AccumulatorType:
        counts, _sum, n_partitions = sparse_acc
        counts = np.where(counts > 0, 1, 0)
        data = None, counts, n_partitions
        self._params.min_sum_per_partition = 0.0
        self._params.max_sum_per_partition = 1.0
        return super().create_accumulator(data)

    def get_sensitivities(self) -> dp_computations.Sensitivities:
        return dp_computations.compute_sensitivities_for_privacy_id_count(
            self._params)


class RawStatisticsCombiner(UtilityAnalysisCombiner):
    """A combiner for computing per-partition raw statistics (count etc)."""
    # (privacy_id_count, count)
    AccumulatorType = Tuple[int, int]

    def create_accumulator(
        self, sparse_acc: Tuple[np.ndarray, np.ndarray,
                                np.ndarray]) -> AccumulatorType:
        count, _sum, n_partitions = sparse_acc
        return len(count), np.sum(count).item()

    def compute_metrics(self, acc: AccumulatorType):
        privacy_id_count, count = acc
        return metrics.RawStatistics(privacy_id_count, count)


class CompoundCombiner(pipeline_dp.combiners.CompoundCombiner):
    """Compound combiner for Utility analysis per partition metrics."""

    # For improving memory usage the compound accumulator has 2 modes:
    # 1. Sparse mode (for small datasets): which contains information about each
    # privacy id's aggregated contributions per partition.
    # 2. Dense mode (for large datasets): which contains accumulators from
    # internal combiners.
    # Since the utility analysis can be run for many configurations, there can
    # be 100s of the internal combiners, as a result the compound
    # accumulator can contain 100s accumulators. Converting each privacy id
    # contribution to such accumulators leads to memory usage blow-up. That is
    # why sparse mode introduced - until the number of privacy id contributions
    # is small, they are saved instead of creating accumulators.
    # In Sparse mode, data (which contains counts, sums, n_partitions) are kept
    # in lists and merge is merging of those lists. For further performance
    # improvements, on converting from sparse to dense mode, the data are
    # converted to NumPy arrays. And internal combiners perform NumPy vector
    # aggregations.
    SparseAccumulatorType = Tuple[List[int], List[float], List[int]]
    DenseAccumulatorType = List[Any]
    AccumulatorType = Tuple[Optional[SparseAccumulatorType],
                            Optional[DenseAccumulatorType]]

    def __init__(self, combiners: Iterable['Combiner'],
                 n_sum_aggregations: int):
        super().__init__(combiners, return_named_tuple=False)
        self._n_sum_aggregations = n_sum_aggregations

    def create_accumulator(self, data: PreaggregatedData) -> AccumulatorType:
        if not data:
            # Handle empty partitions. Only encountered when public partitions
            # are used.
            if self._n_sum_aggregations > 1:
                empty_sum = [(0,) * self._n_sum_aggregations]
            else:
                empty_sum = [0]
            return (([0], empty_sum, [0]), None)
        return (([data[0]], [data[1]], [data[2]]), None)

    def _to_dense(self,
                  sparse_acc: SparseAccumulatorType) -> DenseAccumulatorType:
        # sparse_acc contains lists, convert them to numpy arrays in order to
        # speed up creation of accumulators.
        sparse_acc = [np.array(a) for a in sparse_acc]
        return (
            len(sparse_acc[0]),
            tuple([
                combiner.create_accumulator(sparse_acc)
                for combiner in self._combiners
            ]),
        )

    def _merge_sparse(self, acc1, acc2):
        if acc1 is None:
            return acc2
        if acc2 is None:
            return acc1
        return tuple(_merge_list(s, t) for s, t in zip(acc1, acc2))

    def _merge_dense(self, acc1, acc2):
        if acc1 is None:
            return acc2
        if acc2 is None:
            return acc1
        return super().merge_accumulators(acc1, acc2)

    def merge_accumulators(self, acc1: AccumulatorType, acc2: AccumulatorType):
        sparse1, dense1 = acc1
        sparse2, dense2 = acc2

        sparse_res = self._merge_sparse(sparse1, sparse2)
        merge_res = self._merge_dense(dense1, dense2)
        is_sparse_gr_dense = sparse_res is not None and len(
            sparse_res[0]) > 2 * len(self._combiners)
        if is_sparse_gr_dense:
            merge_res = self._merge_dense(merge_res, self._to_dense(sparse_res))
            sparse_res = None
        return sparse_res, merge_res

        if sparse1 and sparse2:
            merged_sparse = tuple(
                [_merge_list(s, t) for s, t in zip(sparse1, sparse2)])
            # Computes heuristically that the sparse representation is less
            # than dense. For this assume that 1 accumulator is on average
            # has a size of aggregated contributions from 2 privacy ids.
            is_sparse_less_dense = len(
                merged_sparse[0]) <= 2 * len(self._combiners)
            if is_sparse_less_dense:
                return (merged_sparse, None)
            # Dense is smaller, convert to dense.
            return (None, self._to_dense(merged_sparse))
        dense1 = self._to_dense(sparse1) if sparse1 else dense1
        dense2 = self._to_dense(sparse2) if sparse2 else dense2
        merged_dense = super().merge_accumulators(dense1, dense2)
        return (None, merged_dense)

    def compute_metrics(self, acc: AccumulatorType):
        sparse, dense = acc
        if sparse:
            dense = self._merge_dense(dense, self._to_dense(sparse))
        return super().compute_metrics(dense)
