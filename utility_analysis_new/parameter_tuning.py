import pipeline_dp
import utility_analysis_new.dp_engine
from pipeline_dp import pipeline_backend
from pipeline_dp import input_validators
from utility_analysis_new import combiners
from utility_analysis_new import utility_analysis

import dataclasses
from dataclasses import dataclass
import operator
from typing import Callable, List, Optional, Tuple, Union
from enum import Enum
import numpy as np


@dataclass
class FrequencyBin:
    """Represents 1 bin of the histogram.

    The bin represents integers between 'lower' (inclusive) and 'upper'
     (exclusive, not stored in this class, but uniquely determined by 'lower').

    Attributes:
        lower: the lower bound of the bin.
        count: the number of elements in the bin.
        sum: the sum of elements in the bin.
        max: the maximum element in the bin (which might be smaller than upper-1).
    """
    lower: int
    count: int
    sum: int
    max: int

    def __add__(self, other: 'FrequencyBin') -> 'FrequencyBin':
        return FrequencyBin(self.lower, self.count + other.count,
                            self.sum + other.sum, max(self.max, other.max))

    def __eq__(self, other):
        return self.lower == other.lower and self.count == other.count and self.sum == other.sum and self.max == other.max


@dataclass
class Histogram:
    """Represents a histogram over integers."""
    name: str
    bins: List[FrequencyBin]

    def total_count(self):
        return sum([bin.count for bin in self.bins])

    def total_sum(self):
        return sum([bin.sum for bin in self.bins])

    @property
    def max_value(self):
        return self.bins[-1].max

    def quantiles(self, q: List[float]) -> List[int]:
        """Computes approximate quantiles over datasets.
        The output quantiles are chosen only from lower bounds of bins in
        this histogram. For each target quantile q it returns the lower bound of
        the first bin, such that all bins from the left contain not more than
        q part of the data.
        E.g. for quantile 0.8, the returned value is bin.lower for the first
        bin such that the ratio of data in bins to left from 'bin' is <= 0.8.
        Args:
            q: a list of quantiles to compute. It must be sorted in ascending order.
        Returns:
            A list of computed quantiles in the same order as in q.
        """
        assert sorted(q) == q, "Quantiles to compute must be sorted."

        result = []
        total_count_up_to_current_bin = count_smaller = self.total_count()
        i_q = len(q) - 1
        for bin in self.bins[::-1]:
            count_smaller -= bin.count
            ratio_smaller = count_smaller / total_count_up_to_current_bin
            while i_q >= 0 and q[i_q] >= ratio_smaller:
                result.append(bin.lower)
                i_q -= 1
        while i_q >= 0:
            result.append(bin[0].lower)
        return result[::-1]


@dataclass
class ContributionHistograms:
    """Histograms of privacy id contributions."""
    cross_partition_histogram: Histogram
    per_partition_histogram: Histogram


def _to_bin_lower(n: int) -> int:
    """Finds the lower bound of the histogram bin which contains the given integer."""
    # For scalability reasons bins can not be all width=1. For the goals of
    # contribution computations it is ok to have bins of larger values have
    # larger width.
    # Here, the following strategy is used: n is rounded down, such that only 3
    # left-most digits of n is kept, e.g. 123->123, 1234->1230, 12345->12300.
    bound = 1000
    while n > bound:
        bound *= 10

    round_base = bound // 1000
    return n // round_base * round_base


def _compute_frequency_histogram(col, backend: pipeline_backend.PipelineBackend,
                                 name: str):
    """Computes histogram of element frequencies in collection.

    Args:
        col: collection with positive integers.
        backend: PipelineBackend to run operations on the collection.
        name: name which is assigned to the computed histogram.
    Returns:
        1 element collection which contains Histogram.
    """

    col = backend.count_per_element(col, "Frequency of elements")

    # Combiner elements to histogram buckets of increasing sizes. Having buckets
    # of width = 1 is not scalable.
    col = backend.map_tuple(
        col, lambda n, f:
        (_to_bin_lower(n),
         FrequencyBin(lower=_to_bin_lower(n), count=f, sum=f * n, max=n)),
        "To FrequencyBin")

    # (lower_bin_value, FrequencyBin)
    col = backend.combine_per_key(col, operator.add, "Combine FrequencyBins")
    # (lower_bin_value, FrequencyBin)
    col = backend.values(col, "To FrequencyBin")
    # (FrequencyBin)
    col = backend.to_list(col, "To 1 element collection")

    # 1 element collection: [FrequencyBin]

    def bins_to_histogram(bins):
        bins.sort(key=lambda bin: bin.lower)
        return Histogram(name, bins)

    return backend.map(col, bins_to_histogram, "To histogram")


def _list_to_contribution_histograms(
        histograms: List[Histogram]) -> ContributionHistograms:
    """Packs histograms from a list to ContributionHistograms."""
    for histogram in histograms:
        if histogram.name == "CrossPartitionHistogram":
            cross_partition_histogram = histogram
        else:
            per_partition_histogram = histogram
    return ContributionHistograms(cross_partition_histogram,
                                  per_partition_histogram)


def _compute_cross_partition_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of cross partition privacy id contributions.

    This histogram contains: number of privacy ids which contributes to 1 partition,
    to 2 partitions etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
        backend: PipelineBackend to run operations on the collection.

    Returns:
        1 element collection, which contains computed Histogram.
    """

    col = backend.distinct(col, "Distinct (privacy_id, partition_key)")
    # col: (pid, pk)

    col = backend.keys(col, "Drop partition id")
    # col: (pid)

    col = backend.count_per_element(col, "Compute partitions per privacy id")
    # col: (pid, num_pk)

    col = backend.values(col, "Drop privacy id")
    # col: (int)

    return _compute_frequency_histogram(col, backend, "CrossPartitionHistogram")


def _compute_per_partition_histogram(col,
                                     backend: pipeline_backend.PipelineBackend):
    """Computes histogram of per partition privacy id contributions.

    This histogram contains: number of tuple (privacy id, partition_key) which
    have 1 row in datasets, 2 rows etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
        backend: PipelineBackend to run operations on the collection.

    Returns:
        1 element collection, which contains Histogram.
    """
    col = backend.count_per_element(
        col, "Contributions per (privacy_id, partition)")
    # col: ((pid, pk), n)

    col = backend.values(col, "Drop privacy id")
    # col: (int)

    return _compute_frequency_histogram(col, backend, "PerPartitionHistogram")


def compute_contribution_histograms(
        col, data_extractors: pipeline_dp.DataExtractors,
        backend: pipeline_backend.PipelineBackend) -> ContributionHistograms:
    """Computes privacy id contribution histograms."""
    # Extract the columns.
    col = backend.map(
        col, lambda row: (data_extractors.privacy_id_extractor(row),
                          data_extractors.partition_extractor(row)),
        "Extract (privacy_id, partition_key))")
    # col: (pid, pk)

    col = backend.to_multi_transformable_collection(col)
    cross_partition_histogram = _compute_cross_partition_histogram(col, backend)
    # 1 element collection: ContributionHistogram
    per_partition_histogram = _compute_per_partition_histogram(col, backend)
    # 1 element collection: ContributionHistogram
    histograms = backend.flatten(cross_partition_histogram,
                                 per_partition_histogram,
                                 "Histograms to one collection")
    # 2 elements (ContributionHistogram)
    histograms = backend.to_list(histograms, "Histograms to List")
    # 1 element collection: [ContributionHistogram]
    return backend.map(histograms, _list_to_contribution_histograms,
                       "To ContributionHistograms")
    # 1 element (ContributionHistograms)


@dataclass
class UtilityAnalysisRun:
    params: utility_analysis.UtilityAnalysisOptions
    result: combiners.AggregateErrorMetrics


class MinimizingFunction(Enum):
    ABSOLUTE_ERROR = 'absolute_error'
    RELATIVE_ERROR = 'relative_error'


@dataclass
class ParametersToTune:
    """Contains parameters to tune."""
    max_partitions_contributed: bool = False
    max_contributions_per_partition: bool = False
    min_sum_per_partition: bool = False
    max_sum_per_partition: bool = False

    def __post_init__(self):
        if not any(dataclasses.asdict(self).values()):
            raise ValueError("ParametersToTune must have at least 1 parameter "
                             "to tune.")


@dataclass
class TuneOptions:
    """Options for tuning process.

    Note that parameters that are not tuned (e.g. metrics, noise kind) are taken
    from aggregate_params.

    Attributes:
        epsilon, delta: differential privacy budget for aggregations for which
          tuning is performed.
        aggregate_params: parameters of aggregation.
        function_to_minimize: which function of the error to minimize. In case
          if this argument is a callable, it should take 1 argument of type
          AggregateErrorMetrics and return float.
        parameters_to_tune: specifies which parameters to tune.
    """
    epsilon: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams
    function_to_minimize: Union[MinimizingFunction, Callable]
    parameters_to_tune: ParametersToTune

    def __post_init__(self):
        input_validators.validate_epsilon_delta(self.epsilon, self.delta,
                                                "TuneOptions")


@dataclass
class TuneResult:
    """Represents tune results.

    Attributes:
        recommended_params: recommended parameters to use, according to
          minimizing function. Note, that those parameters might not necessarily
          be optimal, since finding the optimal parameters is not always
          feasible.
        options: input options for tuning.
        contribution_histograms: histograms of privacy id contributions.
        utility_analysis_parameters: contains tune parameter values for which
        utility analysis ran.
        index_best: index of the recommended (best) configuration in
        utility_analysis_parameters.
        utility_analysis_results: the results of all utility analysis runs that
          were performed during the tuning process.
    """
    recommended_params: pipeline_dp.AggregateParams
    options: TuneOptions
    contribution_histograms: ContributionHistograms
    utility_analysis_parameters: utility_analysis_new.dp_engine.MultiParameterConfiguration
    index_best: int
    utility_analysis_results: List[combiners.AggregateErrorMetrics]


def _find_candidate_parameters(
    histograms: ContributionHistograms, parameters_to_tune: ParametersToTune
) -> utility_analysis_new.dp_engine.MultiParameterConfiguration:
    # TODO: decide where to put QUANTILES_TO_USE, maybe TuneOptions?
    QUANTILES_TO_USE = [0.9, 0.95, 0.98, 0.99, 0.995]
    l0_candidates = linf_candidates = None

    def _find_candidates(histogram: Histogram) -> List:
        candidates = histogram.quantiles(QUANTILES_TO_USE)
        candidates.append(histogram.max_value)
        candidates = list(set(candidates))  # remove duplicates
        candidates.sort()
        return candidates

    if parameters_to_tune.max_partitions_contributed:
        l0_candidates = _find_candidates(histograms.cross_partition_histogram)

    if parameters_to_tune.max_contributions_per_partition:
        linf_candidates = _find_candidates(histograms.per_partition_histogram)

    l0_bounds = linf_bounds = None

    if l0_candidates and linf_candidates:
        l0_bounds, linf_bounds = [], []
        for l0 in l0_candidates:
            for linf in linf_candidates:
                l0_bounds.append(l0)
                linf_bounds.append(linf)
    elif l0_candidates:
        l0_bounds = l0_candidates
    elif linf_candidates:
        linf_bounds = linf_candidates
    else:
        assert False, "Nothing to tune."

    return utility_analysis_new.dp_engine.MultiParameterConfiguration(
        max_partitions_contributed=l0_bounds,
        max_contributions_per_partition=linf_bounds)


def _utility_analysis_to_tune_result(
        utility_analysis_result: Tuple, tune_options: TuneOptions,
        run_configurations: utility_analysis_new.dp_engine.
    MultiParameterConfiguration, is_public_partitions: bool,
        contribution_histograms: ContributionHistograms) -> TuneResult:

    # Get only error metrics, ignore partition selection for now.
    # TODO: Make the output of the utility analysis 1 dataclass per 1 run.
    if is_public_partitions:
        # utility_analysis_result contains only error metrics.
        aggregate_errors = utility_analysis_result
    else:
        # utility_analysis_result contains partition_selection_metrics,
        # aggregate_errors for each utility run. Extract only aggregate_errors.
        aggregate_errors = utility_analysis_result[1::2]

    assert len(aggregate_errors) == run_configurations.size
    # TODO(dvadym): implement relative error.
    # TODO(dvadym): take into consideration partition selection from private
    # partition selection.
    assert tune_options.function_to_minimize == MinimizingFunction.ABSOLUTE_ERROR

    index_best = np.argmin([ae.absolute_rmse() for ae in aggregate_errors])
    recommended_params = run_configurations.get_aggregate_params(
        tune_options.aggregate_params, index_best)

    return TuneResult(
        recommended_params, tune_options, contribution_histograms,
        run_configurations, index_best,
        utility_analysis_new.dp_engine.MultiParameterConfiguration)


def tune(col,
         backend: pipeline_backend.PipelineBackend,
         contribution_histograms: ContributionHistograms,
         options: TuneOptions,
         data_extractors: pipeline_dp.DataExtractors,
         public_partitions=None) -> TuneResult:
    """Tunes parameters.

    Args:
        col: collection where all elements are of the same type.
          contribution_histograms:
        backend: PipelineBackend with which the utility analysis will be run.
        options: options for tuning.
        data_extractors: functions that extract needed pieces of information
          from elements of 'col'.
        public_partitions: A collection of partition keys that will be present
          in the result. If not provided, tuning will be performed assuming
          private partition selection is used.
    Returns:
        1 element collection which contains TuneResult.
    """
    _check_tune_args(options)

    candidates = _find_candidate_parameters(contribution_histograms,
                                            options.parameters_to_tune)

    utility_analysis_options = utility_analysis.UtilityAnalysisOptions(
        options.epsilon,
        options.delta,
        options.aggregate_params,
        multi_param_configuration=candidates)
    utility_analysis_result = utility_analysis.perform_utility_analysis(
        col, backend, utility_analysis_options, data_extractors,
        public_partitions)
    is_public_partitions = public_partitions is not None
    return backend.map(
        utility_analysis_result,
        lambda result: _utility_analysis_to_tune_result(
            result, options, candidates, is_public_partitions,
            contribution_histograms), "To Tune result")


def _check_tune_args(options: TuneOptions):
    if options.aggregate_params.metrics != [pipeline_dp.Metrics.COUNT]:
        raise NotImplementedError("Tuning only for count is supported.")

    if options.function_to_minimize != MinimizingFunction.ABSOLUTE_ERROR:
        raise NotImplementedError(
            f"Only {MinimizingFunction.ABSOLUTE_ERROR} is implemented.")
