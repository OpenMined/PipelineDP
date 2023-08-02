# Copyright 2023 OpenMined.
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
"""Classes for representing dataset histograms."""

from dataclasses import dataclass
import enum
from typing import List, Sequence, Tuple


@dataclass
class FrequencyBin:
    """Represents a single bin of a histogram.

    The bin represents integers between 'lower' (inclusive) and 'upper'
    (exclusive, not stored in this class, but uniquely determined by 'lower').

    Attributes:
      lower: the lower bound of the bin.
      count: the number of elements in the bin.
      sum: the sum of elements in the bin.
      max: the maximum element in the bin, which is smaller or equal to the
       upper-1.
    """
    lower: int
    count: int
    sum: int
    max: int

    def __add__(self, other: 'FrequencyBin') -> 'FrequencyBin':
        return FrequencyBin(self.lower, self.count + other.count,
                            self.sum + other.sum, max(self.max, other.max))

    def __eq__(self, other):
        return (self.lower == other.lower and self.count == other.count and
                self.sum == other.sum and self.max == other.max)


class HistogramType(enum.Enum):
    # For L0 contribution histogram, for each bin:
    # 'count' is the number of privacy units which contribute to
    # [lower, next_lower) partitions.
    # 'sum' is the total number (privacy_unit, partition) for these privacy
    # units.
    L0_CONTRIBUTIONS = 'l0_contributions'
    L1_CONTRIBUTIONS = 'l1_contributions'
    # For Linf contribution histogram, for each bin:
    # 'count' is the number of pairs (privacy_unit, partition) which contribute
    # with [lower, next_lower) contributions.
    # 'sum' is the total number of contributions for these pairs.
    LINF_CONTRIBUTIONS = 'linf_contributions'
    COUNT_PER_PARTITION = 'count_per_partition'
    COUNT_PRIVACY_ID_PER_PARTITION = 'privacy_id_per_partition_count'


@dataclass
class Histogram:
    """Represents a histogram over integers."""
    name: HistogramType
    bins: List[FrequencyBin]

    def total_count(self):
        return sum([bin.count for bin in self.bins])

    def total_sum(self):
        return sum([bin.sum for bin in self.bins])

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
            q: a list of quantiles to compute. It must be sorted in ascending
            order.

        Returns:
            A list of computed quantiles in the same order as in q.
        """
        assert sorted(q) == q, "Quantiles to compute must be sorted."

        result = []
        total_count_up_to_current_bin = count_smaller = self.total_count()
        if total_count_up_to_current_bin == 0:
            raise ValueError("Cannot compute quantiles of an empty histogram")
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


def compute_ratio_dropped(
        contribution_histogram: Histogram) -> Sequence[Tuple[int, float]]:
    """Computes ratio of dropped data for different bounding thresholds.

    For each FrequencyBin.lower in contribution_histogram it computes what would
    the ratio of data dropped because of contribution bounding when it is taken
    as bounding threshold (e.g. in case of L0 histogram bounding_threshold it is
    max_partition_contribution). For convenience the (0, 1) is added as 1st
    element.

    Args:
        contribution_histogram: histogram of contributions. It can be L0, L1,
          Linf contribution bounding histogram.

    Returns:
        A sequence of sorted pairs
        (bounding_threshold:int, ratio_data_draped:float), where
        bounding_thresholds are all lower of histogram.bins and
        contribution_histogram.max_value.
    """
    if not contribution_histogram.bins:
        return []
    dropped = elements_larger = 0
    total_sum = contribution_histogram.total_sum()
    ratio_dropped = []
    bins = contribution_histogram.bins
    previous_value = bins[-1].lower  # lower of the largest bin.
    if contribution_histogram.max_value() != previous_value:
        # Add ratio for max_value when max_value is not lower in bins.
        ratio_dropped.append((contribution_histogram.max_value(), 0.0))

    for bin in bins[::-1]:
        current_value = bin.lower
        dropped += elements_larger * (previous_value - current_value) + (
            bin.sum - bin.count * current_value)
        ratio_dropped.append((current_value, dropped / total_sum))
        previous_value = current_value
        elements_larger += bin.count
    ratio_dropped.append((0, 1))
    return ratio_dropped[::-1]


@dataclass
class DatasetHistograms:
    """Contains histograms useful for parameter tuning."""
    l0_contributions_histogram: Histogram
    l1_contributions_histogram: Histogram
    linf_contributions_histogram: Histogram
    count_per_partition_histogram: Histogram
    count_privacy_id_per_partition: Histogram
