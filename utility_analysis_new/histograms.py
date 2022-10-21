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

import pipeline_dp
from pipeline_dp import pipeline_backend
from dataclasses import dataclass
import operator
from typing import List


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
    col = backend.reduce_per_key(col, operator.add, "Combine FrequencyBins")
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