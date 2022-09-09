import pipeline_dp
from pipeline_dp import pipeline_backend
from dataclasses import dataclass
import operator
from typing import Iterable, List


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
class ContributionHistogram:  # name?
    name: str
    bins: List[FrequencyBin]


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
    """Computes histogram of elements frequencies in collection.

    Args:
        col: collection with positive integers.
        backend: PipelineBackend to run operations on the collection.
        name: todo
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
    # (FrequencyBin,)
    col = backend.to_list(col, "To 1 element collection")

    # 1 element collection: [FrequencyBin]

    def bins_to_histogram(bins):
        bins.sort(key=lambda bin: bin.lower)
        return ContributionHistogram(name, bins)

    return backend.map(col, bins_to_histogram, "To histogram")


@dataclass
class ContributionHistograms:
    cross_partition_histogram: ContributionHistogram
    per_partition_histogram: ContributionHistogram
    # todo: add sum_histograms


def _list_to_contribution_histograms(histograms: List[ContributionHistogram]):
    cross_partition_histogram = per_partitionn_histogram = None
    for histogram in histograms:
        if histogram.name == "CrossPartitionHistogram":
            cross_partition_histogram = histogram
        else:
            per_partition_histogram = histogram
    return ContributionHistograms(cross_partition_histogram,
                                  per_partition_histogram)


def _compute_cross_partition_histogram(
        col, backend: pipeline_backend.PipelineBackend):

    def count_unique_elements(elements: Iterable):
        unique = set()
        for partition in elements:
            unique.add(partition)
        return len(unique)

    col = backend.group_by_key(col, "Group by privacy_id")
    # col: (pid, [pk])
    col = backend.values(col, "Drop privacy id")
    # col: ([pk])
    col = backend.map(col, count_unique_elements,
                      "Count partition keys per privacy_id")
    # col: (int)
    return _compute_frequency_histogram(col, backend, "CrossPartitionHistogram")


def _compute_per_partition_histogram(col,
                                     backend: pipeline_backend.PipelineBackend):
    col = backend.count_per_element(
        col, "Contributions per (privacy_id, partition)")
    # col: ((pid, pk), n)

    col = backend.values(col, "Drop privacy id")
    # col: (int)

    return _compute_frequency_histogram(col, backend, "PerPartitionHistogram")


def compute_contribution_histograms(
        col, data_extractors: pipeline_dp.DataExtractors,
        backend: pipeline_backend.PipelineBackend) -> ContributionHistograms:
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
    histograms = backend.map(histograms, _list_to_contribution_histograms,
                             "To ContributionHistograms")
    # 1 element (ContributionHistograms)
    return histograms
