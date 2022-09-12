import pipeline_dp
from dataclasses import dataclass
import operator
from typing import Callable, Generic, List, Optional, TypeVar, Union
from utility_analysis_new import combiners
from utility_analysis_new import utility_analysis
from enum import Enum


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


def _compute_frequency_histogram(
        col, backend: pipeline_dp.pipeline_backend.PipelineBackend):
    """Computes histogram of element frequencies in collection.

    Args:
        col: collection with positive integers.
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains a list of FrequencyBin sorted by
        'lower' attribute in ascending order.
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

    def sort_histogram(bins):
        bins.sort(key=lambda bin: bin.lower)
        return bins

    return backend.map(col, sort_histogram, "Sort histogram")


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
    """Defines restrictions for numbers (int or floats)."""
    lower: Optional[T] = None
    upper: Optional[T] = None

    def __post_init__(self):
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
        epsilon, delta: differential private budget for aggregations for which
        tuning is performed.
        aggregate_params: parameters of aggregation.
        function_to_minimize: which function of the error to minimize. In case
        if this argument is callable, it should take 1 argument of type
        AggregateErrorMetrics and return float.
        parameters_to_tune: spefies which parameters are tunable and what
        restrictions on their values.
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
        minimizing function. Note, that those parameters might be not optimal,
        since finding optimal parameters not always feasible.
        options: input options of tuning (for convenience).
        contribution_histograms: histograms of privacy id contributions.
        utility_analysis_runs: the results of all utility analysis runs that
        were performed during tuning process.
    """
    recommended_params: pipeline_dp.AggregateParams
    options: TuneOptions
    contribution_histograms: ContributionHistograms
    utility_analysis_runs: List[UtilityAnalysisRun]


def perform_parameter_tuning(col,
                             contribution_histograms: ContributionHistograms,
                             options: TuneOptions,
                             data_extractors: pipeline_dp.DataExtractors,
                             public_partitions=None) -> TuneResult:
    pass
