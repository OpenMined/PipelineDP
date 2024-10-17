# Copyright 2024 OpenMined.
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
"""Functions for computing linf_sum and sum_per_partition histograms."""

# This file contains histograms which are useful for analysis of DP SUM
# aggregation utility.
# The general structure of these histogram is the following:
# The input is a collection of values X = (x_0, ... x_n).
# The computations are as follows:
# 1. Find min_x = min(X), max_x = max(X) of X
# 2. Split the segment [min_x, max_x] in NUMBER_OF_BUCKETS_SUM_HISTOGRAM = 10000
# equal size intervals [l_i, r_i). The last internal includes max_x.
# 3. Each bin of the histogram corresponds to interval [l_i, r_i), and contains
# different statistics (count, sum etc) of numbers from X, which lies in this
# interval.
#
# For generating bucket class LowerUpperGenerator is used, which takes
# min, max, number of buckets and returns bucket for each number.

import copy
import operator
from typing import Iterable, List, Tuple

from pipeline_dp import pipeline_backend, pipeline_functions
from pipeline_dp.dataset_histograms import histograms as hist

NUMBER_OF_BUCKETS = 10000


class LowerUpperGenerator:
    """Generates lower&upper bounds for FrequencyBin

    Attributes:
        left, right: bounds on interval on which we compute histogram.
        num_buckets: number of buckets in [left, right]. Buckets have the same
         length.

    For general context see file docstring.
    i-th bucket corresponds to numbers from
    [left+i*bucket_len, right+(i+1)*bucket_len), where
    bucket_len = (right-left)/num_buckets.
    The last bucket includes right end-point.
    """

    def __init__(
        self,
        left: float,
        right: float,
        num_buckets: int = NUMBER_OF_BUCKETS,
    ):
        assert left <= right, "The left bound must be smaller then the right one, but {left=} and {right=}"
        self.left = left
        self.right = right
        self.num_buckets = num_buckets if left < right else 1
        self.bucket_len = (right - left) / num_buckets

    def get_bucket_index(self, x: float) -> int:
        if x >= self.right:  # last bucket includes both ends.
            return self.num_buckets - 1
        if x <= self.left:
            return 0
        if x >= self.right:
            return self.num_buckets - 1
        return int((x - self.left) / self.bucket_len)

    def get_lower_upper(self, x: float) -> Tuple[float, float]:
        index = self.get_bucket_index(x)
        return self._get_lower(index), self._get_upper(index)

    def _get_lower(self, index: int) -> float:
        return self.left + index * self.bucket_len

    def _get_upper(self, index: int) -> float:
        return self.left + (index + 1) * self.bucket_len


def _compute_frequency_histogram_per_key(
    col,
    backend: pipeline_backend.PipelineBackend,
    name: hist.HistogramType,
    num_buckets: int,
):
    """Computes histogram of element frequencies in collection.

    This is a helper function for computing sum histograms per key.

    Args:
      col: collection of (key, value:float)
      backend: PipelineBackend to run operations on the collection.
      name: name which is assigned to the computed histogram.
      num_buckets: the number of buckets in the output histogram.

    Returns:
      1 element collection which contains list [hist.Histogram], sorted by key.
    """
    col = backend.to_multi_transformable_collection(col)

    bucket_generators = _create_bucket_generators_per_key(
        col, num_buckets, backend)

    def _map_to_frequency_bin(
        key_value: Tuple[int, float],
        bucket_generators: List[List[LowerUpperGenerator]]
    ) -> Tuple[Tuple[int, float], hist.FrequencyBin]:
        # bucket_generator is a 1-element list with
        # a single element to be a list of LowerUpperGenerator.
        index, value = key_value
        bucket_generator = bucket_generators[0][index]
        bin_lower, bin_upper = bucket_generator.get_lower_upper(value)
        bucket = hist.FrequencyBin(lower=bin_lower,
                                   upper=bin_upper,
                                   count=1,
                                   sum=value,
                                   max=value,
                                   min=value)
        return (index, bin_lower), bucket

    col = backend.map_with_side_inputs(col, _map_to_frequency_bin,
                                       (bucket_generators,), "To FrequencyBin")
    # (lower_bin_value, hist.FrequencyBin)

    col = backend.reduce_per_key(col, operator.add, "Combine FrequencyBins")
    # ((index, lower_bin_value), hist.FrequencyBin)

    col = backend.map_tuple(col, lambda k, v: (k[0], v), "Drop lower")
    # (index, hist.FrequencyBin)

    col = backend.group_by_key(col, "Group by histogram index")

    # (index, [hist.FrequencyBin])

    def bins_to_histogram(bins):
        sorted_bins = sorted(bins, key=lambda bin: bin.lower)
        return hist.Histogram(name, sorted_bins)

    col = backend.map_values(col, bins_to_histogram, "To histogram")

    col = backend.to_list(col, "To 1 element collection")

    def sort_histograms_by_index(index_histogram):
        if len(index_histogram) == 1:
            # It is a histogram for one column, return it w/o putting it in a list.
            return index_histogram[0][1]

        # Sort histograms by index and return them as a list.
        # Beam does not like mutating arguments, so copy the argument.
        index_histogram = copy.deepcopy(index_histogram)
        return [histogram for index, histogram in sorted(index_histogram)]

    col = backend.map(col, sort_histograms_by_index, "sort histogram by index")
    # 1 element collection with list of histograms: [hist.FrequencyBin]
    return col


def _create_bucket_generators_per_key(
        col, number_of_buckets: int, backend: pipeline_backend.PipelineBackend):
    """Creates bucket generators per key.

    Args:
        col: collection of (key, value)
        backend: PipelineBackend to run operations on the collection.
        num_buckets: the number of buckets in the output histogram.

    Returns:
        1 element collection with dictionary {key: LowerUpperGenerator}.
    """
    col = pipeline_functions.min_max_per_key(
        backend, col, "Min and max value per value column")
    # (index, (min, max))

    col = backend.to_list(col, "To list")

    # 1 elem col: ([(index, (min, max))])

    def create_generators(index_min_max: List[Tuple[int, Tuple[float, float]]]):
        min_max_sorted_by_index = [v for k, v in sorted(index_min_max)]
        return [
            LowerUpperGenerator(min, max, number_of_buckets)
            for min, max in min_max_sorted_by_index
        ]

    return backend.map(col, create_generators, "Create generators")


def _flat_values(col, backend: pipeline_backend.PipelineBackend):
    """Unnests values in (key, value) collection.

    Args:
        col: collection of (key, value) or (key, [value])
        backend: PipelineBackend to run operations on the collection.

    Transform each element:
    (key, value: float) -> ((0, key), value)
    (key, value: list[float]) -> [((0, key), value[0]), ((1, key), value[1])...]
    and then unnest them.

    Return:
        Collection of ((index, key), value).
    """

    def flat_values(key_values):
        key, values = key_values
        if isinstance(values, Iterable):
            for i, value in enumerate(values):
                yield (i, key), value
        else:
            yield (0, key), values  # 1 value

    return backend.flat_map(col, flat_values, "Flat values")


def _compute_linf_sum_contributions_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of per partition privacy id contributions.

    This histogram contains: the number of (privacy id, partition_key)-pairs
    which have sum of values X_1, X_2, ..., X_n, where X_1 = min_sum,
    X_n = one before max sum and n is equal to
    NUMBER_OF_BUCKETS_SUM_HISTOGRAM.

    Args:
        col: collection with elements ((privacy_id, partition_key), value(s)).
         Where value(s) can be one float of list of floats.
        where value can be 1 float or tuple of floats (in case of many columns)
        backend: PipelineBackend to run operations on the collection.

    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """
    col = _flat_values(col, backend)
    # ((index_value, (pid, pk)), value).
    col = backend.sum_per_key(
        col, "Sum of contributions per (privacy_id, partition)")
    # col: ((index, (pid, pk), sum_per_key)
    col = backend.map_tuple(col, lambda k, v: (k[0], v),
                            "Drop privacy_id, partition_key")
    # col: (index, float)

    return _compute_frequency_histogram_per_key(
        col, backend, hist.HistogramType.LINF_SUM_CONTRIBUTIONS,
        NUMBER_OF_BUCKETS)


def _compute_partition_sum_histogram(col,
                                     backend: pipeline_backend.PipelineBackend):
    """Computes histogram of sum per partition.

    This histogram contains: the number of partition_keys which have sum of
    values X_1, X_2, ..., X_n, where X_1 = min_sum, X_n = one before max sum and
    n is equal to NUMBER_OF_BUCKETS_SUM_HISTOGRAM.

    Args:
      col: collection with elements ((privacy_id, partition_key), value).
      backend: PipelineBackend to run operations on the collection.
    Returns:
          1 element collection, which contains the computed hist.Histogram.
    """

    col = backend.map_tuple(col, lambda pid_pk, value: (pid_pk[1], value),
                            "Drop privacy id")
    # (pk, values)
    col = _flat_values(col, backend)
    # ((index_value, pk), value).
    col = backend.sum_per_key(col, "Sum of contributions per partition")
    # col: ((index_value, pk), sum_per_partition)
    col = backend.map_tuple(col, lambda index_pk, value: (index_pk[0], value),
                            "Drop partition")
    # col: (index, float)
    return _compute_frequency_histogram_per_key(
        col, backend, hist.HistogramType.SUM_PER_PARTITION, NUMBER_OF_BUCKETS)


def _compute_linf_sum_contributions_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of per partition privacy id contributions.

    This histogram contains: the number of (privacy id, partition_key)-pairs
    which have sum of values X_1, X_2, ..., X_n, where X_1 = min_sum,
    X_n = one before max sum and n is equal to
    NUMBER_OF_BUCKETS_SUM_HISTOGRAM.

    Args:
      col: collection with a pre-aggregated dataset, each element is
      (partition_key, (count, sum, n_partitions, n_contributions)).
      backend: PipelineBackend to run operations on the collection.
    Returns:
      1 element collection, which contains the computed histograms.Histogram.
    """
    col = backend.map_tuple(
        col,
        lambda _, x:
        (None, x[1]),  # x is (count, sum, n_partitions, n_contributions)
        "Extract sum per partition contribution")
    # col: (values,) where each element is the sum of values the todo
    # corresponding privacy_id contributes to the partition.

    col = _flat_values(col, backend)
    # col: ((index, None), float)

    col = backend.map_tuple(col, lambda k, v: (k[0], v), "Drop dummy key")
    # col: (index, float)

    return _compute_frequency_histogram_per_key(
        col, backend, hist.HistogramType.LINF_SUM_CONTRIBUTIONS,
        NUMBER_OF_BUCKETS)


def _compute_partition_sum_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of counts per partition.

    This histogram contains: the number of partition_keys which have sum of
    values X_1, X_2, ..., X_n, where X_1 = min_sum, X_n = one before max sum and
    n is equal to NUMBER_OF_BUCKETS_SUM_HISTOGRAM.

    Args:
      col: collection with a pre-aggregated dataset, each element is
      (partition_key, (count, sum, n_partitions, n_contributions)).
      backend: PipelineBackend to run operations on the collection.
    Returns:
      1 element collection, which contains the computed histograms.Histogram.g
    """
    col = backend.map_values(
        col,
        lambda x: x[1],  # x is (count, sum, n_partitions, n_contributions)
        "Extract sum per partition contribution")
    # col: (pk, int)

    col = _flat_values(col, backend)
    # col: ((index, pk), float)

    col = backend.sum_per_key(col, "Sum per partition")
    # col: ((index, pk), float), where each element is the total sum per partition.
    col = backend.map_tuple(col, lambda k, v: (k[0], v),
                            "Drop privacy_id, partition_key")
    # col: (index, float)

    return _compute_frequency_histogram_per_key(
        col, backend, hist.HistogramType.SUM_PER_PARTITION, NUMBER_OF_BUCKETS)
