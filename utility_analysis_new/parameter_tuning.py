import pipeline_dp
from pipeline_dp import pipeline_backend
from dataclasses import dataclass
import operator
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union
from utility_analysis_new import combiners
from utility_analysis_new import utility_analysis
from enum import Enum
import numbers


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
class ContributionHistograms:
    """Histograms of privacy id contributions."""
    cross_partition_histogram: Histogram
    per_partition_histogram: Optional[Histogram] = None


@dataclass
class UtilityAnalysisRun:
    params: utility_analysis.UtilityAnalysisOptions
    result: combiners.AggregateErrorMetrics


class MinimizingFunction(Enum):
    ABSOLUTE_ERROR = 'absolute_error'
    RELATIVE_ERROR = 'relative_error'


T = TypeVar('T')


@dataclass
class Bounds(Generic[T]):
    """Defines optional lower and upper bounds (int or float)."""
    lower: Optional[T] = None
    upper: Optional[T] = None

    def __post_init__(self):

        def check_is_number(value, name: str):
            if name is None:
                return
            if not isinstance(value, numbers.Number):
                raise ValueError(f"{name} must be number, but {name}={value}")

        check_is_number(self.lower, "lower")
        check_is_number(self.upper, "upper")

        if self.lower is not None and self.upper is not None:
            assert self.lower <= self.upper


@dataclass
class TunedParameters:
    """Contains parameters to tune.

    Attributes of this class define restrictions on tuning on corresponding
    attributes in pipeline_dp.AggregateParams.

    Example:
        max_partitions_contributed
          1. = Bound(lower=1, upper=10) only numbers between 1 and 10 will be
          considered during tuning for max_partitions_contributed.
          2. = Bound(lower=None, upper=10) means numbers till 10.
          3. = None means no restriction
          4. = 3 means no tuning, max_partitions_contributed=3.
    """
    max_partitions_contributed: Optional[Union[int, Bounds[int]]] = None
    max_contributions_per_partition: Optional[Union[int, Bounds[int]]] = None
    min_sum_per_partition: Optional[Union[float, Bounds[float]]] = None
    max_sum_per_partition: Optional[Union[float, Bounds[float]]] = None

    def __post_init__(self):

        def check_int_attribute(value, name: str):
            if value is None:
                return
            if isinstance(value, int):
                if value <= 0:
                    raise ValueError(f"{name} must be >0, but {name}={value}")
            elif isinstance(value, Bounds):
                if value.lower <= 0:
                    raise ValueError(
                        f"{name} lower bound must be >0, but {name}.lower={value}"
                    )

        check_int_attribute(self.max_partitions_contributed,
                            "max_partitions_contributed")
        check_int_attribute(self.max_contributions_per_partition,
                            "max_contributions_per_partition")


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
        parameters_to_tune: specifies which parameters are tunable and with
          optional restrictions on their values.
    """
    epsilon: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams
    function_to_minimize: Union[MinimizingFunction, Callable]
    parameters_to_tune: TunedParameters


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
        utility_analysis_runs: the results of all utility analysis runs that
          were performed during the tuning process.
    """
    recommended_params: pipeline_dp.AggregateParams
    options: TuneOptions
    contribution_histograms: ContributionHistograms
    utility_analysis_runs: List[UtilityAnalysisRun]


def tune_parameters(col,
                    contribution_histograms: ContributionHistograms,
                    options: TuneOptions,
                    data_extractors: pipeline_dp.DataExtractors,
                    public_partitions=None) -> TuneResult:
    """Tunes parameters.

    Args:
        col: collection where all elements are of the same type.
          contribution_histograms:
        options: options for tuning.
        data_extractors: functions that extract needed pieces of information
          from elements of 'col'.
        public_partitions: A collection of partition keys that will be present
          in the result. If not provided, tuning will be performed assuming
          private partition selection is used.
    """
    raise NotImplementedError("tune_parameters is not yet implemented.")
