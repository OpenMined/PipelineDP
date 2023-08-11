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
"""Functions for computing dataset histograms in pipelines."""
import bisect
import operator
from typing import List, Tuple

import numpy

import pipeline_dp
from pipeline_dp import pipeline_backend, pipeline_functions
from pipeline_dp.dataset_histograms import histograms as hist

NUMBER_OF_BUCKETS_IN_LINF_SUM_CONTRIBUTIONS_HISTOGRAM = 10000


def _to_bin_lower_logarithmic(value: int) -> int:
    """Finds the lower bound of the histogram bin which contains the given integer.

    Keep in sync with private_contribution_bounds._generate_possible_contribution_bounds.
    """
    # For scalability reasons bins can not be all width=1. For the goals of
    # contribution computations it is ok to have bins of larger values have
    # larger width.
    # Here, the following strategy is used: n is rounded down, such that only 3
    # left-most digits of n is kept, e.g. 123->123, 1234->1230, 12345->12300.
    bound = 1000
    while value > bound:
        bound *= 10

    round_base = bound // 1000
    return value // round_base * round_base


def _to_bin_lower_with_lowers(lowers: [float], value: float) -> float:
    """Finds the lower bound of the histogram bin which contains the given number."""
    bin_lower_idx = bisect.bisect_right(lowers, value) - 1
    assert bin_lower_idx >= 0
    return lowers[bin_lower_idx]


def _compute_frequency_histogram(col, backend: pipeline_backend.PipelineBackend,
                                 name: hist.HistogramType):
    """Computes histogram of element frequencies in collection.

    Args:
        col: collection with positive integers.
        backend: PipelineBackend to run operations on the collection.
        name: name which is assigned to the computed histogram.
    Returns:
        1 element collection which contains hist.Histogram.
    """

    col = backend.count_per_element(col, "Frequency of elements")

    # Combiner elements to histogram buckets of increasing sizes. Having buckets
    # of width = 1 is not scalable.
    return _compute_frequency_histogram_helper(backend, col, name)


def _compute_weighted_frequency_histogram(
        col, backend: pipeline_backend.PipelineBackend,
        name: hist.HistogramType):
    """Computes histogram of element frequencies in collection.

    Args:
        col: collection of (positive integers, weight).
        backend: PipelineBackend to run operations on the collection.
        name: name which is assigned to the computed histogram.
    Returns:
        1 element collection which contains hist.Histogram.
  """

    col = backend.sum_per_key(col, "Frequency of elements")
    # (int, sum_weights:float)

    col = backend.map_values(col, lambda x: int(round(x)), "Round")
    # (int, sum_weights:int)

    # Combiner elements to histogram buckets of increasing sizes. Having buckets
    # of width = 1 is not scalable.
    return _compute_frequency_histogram_helper(backend, col, name)


def _compute_frequency_histogram_helper(
        backend: pipeline_backend.PipelineBackend, col,
        name: hist.HistogramType):
    """Computes histogram of element frequencies in collection.

    This is a helper function for _compute_frequency_histogram and
    _compute_weighted_frequency_histogram.

    Args:
        col: collection of (n:int, frequency_of_n: int)
        backend: PipelineBackend to run operations on the collection.
        name: name which is assigned to the computed histogram.
    Returns:
        1 element collection which contains hist.Histogram.
    """

    def _map_to_frequency_bin(value: int,
                              frequency: int) -> Tuple[int, hist.FrequencyBin]:
        bin_lower = _to_bin_lower_logarithmic(value)
        return bin_lower, hist.FrequencyBin(lower=bin_lower,
                                            count=frequency,
                                            sum=frequency * value,
                                            max=value)

    col = backend.map_tuple(col, _map_to_frequency_bin, "To FrequencyBin")
    # (lower_bin_value, hist.FrequencyBin)
    return _convert_frequency_bins_into_histogram(backend, col, name)


def _compute_frequency_histogram_helper_with_lowers(
        backend: pipeline_backend.PipelineBackend, col,
        name: hist.HistogramType, lowers_col):
    """Computes histogram of element frequencies in collection.

    This is a helper function for _compute_frequency_histogram and
    _compute_weighted_frequency_histogram.

    Args:
        col: collection of (n:int, frequency_of_n: int)
        backend: PipelineBackend to run operations on the collection.
        name: name which is assigned to the computed histogram.
    Returns:
        1 element collection which contains hist.Histogram.
    """

    def _map_to_frequency_bin(
            value: float, frequency: int,
            lowers: List[float]) -> Tuple[float, hist.FrequencyBin]:
        bin_lower = _to_bin_lower_with_lowers(lowers, value)
        return bin_lower, hist.FrequencyBin(lower=bin_lower,
                                            count=frequency,
                                            sum=frequency * value,
                                            max=value)

    col = backend.map_tuple_with_side_inputs(col, _map_to_frequency_bin,
                                             (lowers_col,), "To FrequencyBin")
    # (lower_bin_value, hist.FrequencyBin)

    return _convert_frequency_bins_into_histogram(backend, col, name)


def _convert_frequency_bins_into_histogram(
        backend: pipeline_backend.PipelineBackend, col, name):
    # (lower_bin_value, hist.FrequencyBin)
    col = backend.reduce_per_key(col, operator.add, "Combine FrequencyBins")
    # (lower_bin_value, hist.FrequencyBin)
    col = backend.values(col, "Drop keys")
    # (hist.FrequencyBin)
    col = backend.to_list(col, "To 1 element collection")

    # 1 element collection: [hist.FrequencyBin]

    def bins_to_histogram(bins):
        sorted_bins = sorted(bins, key=lambda bin: bin.lower)
        return hist.Histogram(name, sorted_bins)

    return backend.map(col, bins_to_histogram, "To histogram")


def _list_to_contribution_histograms(
        histograms: List[hist.Histogram]) -> hist.DatasetHistograms:
    """Packs histograms from a list to ContributionHistograms."""
    l0_contributions = l1_contributions = linf_contributions = None
    count_per_partition = privacy_id_per_partition_count = None
    for histogram in histograms:
        if histogram.name == hist.HistogramType.L0_CONTRIBUTIONS:
            l0_contributions = histogram
        if histogram.name == hist.HistogramType.L1_CONTRIBUTIONS:
            l1_contributions = histogram
        elif histogram.name == hist.HistogramType.LINF_CONTRIBUTIONS:
            linf_contributions = histogram
        elif histogram.name == hist.HistogramType.COUNT_PER_PARTITION:
            count_per_partition = histogram
        elif histogram.name == hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION:
            privacy_id_per_partition_count = histogram
    return hist.DatasetHistograms(l0_contributions, l1_contributions,
                                  linf_contributions, count_per_partition,
                                  privacy_id_per_partition_count)


def _to_dataset_histograms(histogram_list,
                           backend: pipeline_backend.PipelineBackend):
    """Combines histogram_list to hist.DatasetHistograms."""
    histograms = backend.flatten(histogram_list, "Histograms to one collection")
    # histograms: 4 elements collection with elements ContributionHistogram

    histograms = backend.to_list(histograms, "Histograms to List")
    # 1 element collection: [ContributionHistogram]
    return backend.map(histograms, _list_to_contribution_histograms,
                       "To ContributionHistograms")
    # 1 element collection: (hist.DatasetHistograms)


############## Computing histograms on raw datasets ##########################
def _compute_l0_contributions_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of the number of distinct partitions contributed by a privacy id.

    This histogram contains: number of privacy ids which contributes 1 record, 2
      records, etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
          Assumption: 'col' contains distinct elements!
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """
    col = backend.keys(col, "Drop partition id")
    # col: (pid)

    col = backend.count_per_element(col, "Compute partitions per privacy id")
    # col: (pid, num_pk)

    col = backend.values(col, "Drop privacy id")
    # col: (int)

    return _compute_frequency_histogram(col, backend,
                                        hist.HistogramType.L0_CONTRIBUTIONS)


def _compute_l1_contributions_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of the number of distinct records contributed by a privacy id.

    This histogram contains: number of privacy ids which contributes to 1
    record, to 2 records etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """
    col = backend.keys(col, "Drop partition id")
    # col: (pid)

    col = backend.count_per_element(col, "Compute records per privacy id")
    # col: (pid, num_records)

    col = backend.values(col, "Drop privacy id")
    # col: (int)

    return _compute_frequency_histogram(col, backend,
                                        hist.HistogramType.L1_CONTRIBUTIONS)


def _compute_linf_contributions_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of per partition privacy id contributions.

    This histogram contains: the number of (privacy id, partition_key)-pairs
    which have 1 row in the datasets, 2 rows etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """
    col = backend.count_per_element(
        col, "Contributions per (privacy_id, partition)")
    # col: ((pid, pk), n)

    col = backend.values(col, "Drop privacy id")
    # col: (int)

    return _compute_frequency_histogram(col, backend,
                                        hist.HistogramType.LINF_CONTRIBUTIONS)


def _compute_linf_sum_contributions_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of per partition privacy id contributions.

    This histogram contains: the number of (privacy id, partition_key)-pairs
    which have sum of values X_1, X_2, ..., X_n, where X_1 = min_sum,
    X_n = one before max sum and n is a constant hardcoded in code.

    Args:
        col: collection with elements (privacy_id, partition_key, value).
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """
    col = backend.sum_per_key(
        col, "Sum of contributions per (privacy_id, partition)")
    # col: ((pid, pk), sum_per_key)
    col = backend.values(col, "Drop keys")
    # col: (float)
    min_max_values = pipeline_functions.min_max_elements(
        backend, col, "Min and max value in dataset")
    lowers = backend.map(
        min_max_values, lambda min_max: numpy.linspace(min_max[0], min_max[1], (
            NUMBER_OF_BUCKETS_IN_LINF_SUM_CONTRIBUTIONS_HISTOGRAM + 1))[:-1],
        "Map to lowers")
    lowers = backend.flatten(lowers, "Flatten lowers")

    return _compute_frequency_histogram_helper_with_lowers(
        col, backend, hist.HistogramType.LINF_CONTRIBUTIONS, lowers)


def _compute_partition_count_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of counts per partition.

    This histogram contains: the number of partitions with total count of
    contributions = 1, 2 etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """

    col = backend.values(col, "Drop privacy keys")
    # col: (pk)

    col = backend.count_per_element(col, "Count per partition")
    # col: (pk, count)

    col = backend.values(col, "Drop partition key")
    # col: (count)

    return _compute_frequency_histogram(col, backend,
                                        hist.HistogramType.COUNT_PER_PARTITION)


def _compute_partition_privacy_id_count_histogram(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of privacy id counts per partition.

    This histogram contains: the number of partitions with privacy_id_count=1,
    with privacy_id_count=2 etc.

    Args:
        col: collection with elements (privacy_id, partition_key).
         Assumption: 'col' contains distinct elements!
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed hist.Histogram.
    """

    col = backend.values(col, "Drop privacy key")
    # col: (pk)

    col = backend.count_per_element(col, "Compute partitions per privacy id")
    # col: (pk, count_pid_per_pk)

    col = backend.values(col, "Drop partition key")
    # col: (int)

    return _compute_frequency_histogram(
        col, backend, hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION)


def compute_dataset_histograms(col, data_extractors: pipeline_dp.DataExtractors,
                               backend: pipeline_backend.PipelineBackend):
    """Computes dataset histograms.

    Args:
        col: collection with elements of the same type.
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains a histograms.DatasetHistograms
        object.
    """

    # Extract the columns.
    col_with_values = backend.map(
        col, lambda row: ((data_extractors.privacy_id_extractor(row),
                           data_extractors.partition_extractor(row)),
                          data_extractors.value_extractor(row)),
        "Extract ((privacy_id, partition_key), value)")
    # col: ((pid, pk), value)

    col_with_values = backend.to_multi_transformable_collection(col_with_values)
    # col: ((pid, pk), value)

    col = backend.keys(col_with_values, "Drop values")
    # col: (pid, pk)

    col = backend.to_multi_transformable_collection(col)
    # col: (pid, pk)

    col_distinct = backend.distinct(col, "Distinct (privacy_id, partition_key)")
    # col: (pid, pk)

    col_distinct = backend.to_multi_transformable_collection(col_distinct)
    # col: (pid, pk)

    # Compute histograms.
    l0_contributions_histogram = _compute_l0_contributions_histogram(
        col_distinct, backend)
    l1_contributions_histogram = _compute_l1_contributions_histogram(
        col, backend)
    linf_contributions_histogram = _compute_linf_contributions_histogram(
        col, backend)
    linf_sum_contributions_histogram = _compute_linf_sum_contributions_histogram(
        col_with_values, backend)
    partition_count_histogram = _compute_partition_count_histogram(col, backend)
    partition_privacy_id_count_histogram = _compute_partition_privacy_id_count_histogram(
        col_distinct, backend)
    # all histograms are 1 element collections which contains ContributionHistogram

    # Combine histograms to histograms.DatasetHistograms.
    return _to_dataset_histograms([
        l0_contributions_histogram, l1_contributions_histogram,
        linf_contributions_histogram, partition_count_histogram,
        partition_privacy_id_count_histogram
    ], backend)


########## Computing histograms on pre-aggregated datasets ####################
# More details on pre-aggregate datatests are in the docstring of function
# pre_aggregation.preaggregate.


def _compute_l0_contributions_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of the number of distinct partitions contributed by a privacy id.

    This histogram contains: number of privacy ids which contributes to 1
    partition, to 2 partitions etc.

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
        (x[2], 1.0 / x[2]),  # x is (count, sum, n_partitions, n_contributions)
        "Extract n_partitions")
    # col: (int,), where each element is the number of partitions the
    # corresponding privacy_id contributes.
    return _compute_weighted_frequency_histogram(
        col, backend, hist.HistogramType.L0_CONTRIBUTIONS)


def _compute_l1_contributions_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of the number of distinct partitions contributed by a privacy id.

    This histogram contains: number of privacy ids which contributes to 1
    partition, to 2 partitions etc.

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
        (x[3], 1 / x[2]),  # x is (count, sum, n_partitions, n_contributions)
        "Extract n_partitions")
    # col: (int,), where each element is the number of partitions the
    # corresponding privacy_id contributes.
    return _compute_weighted_frequency_histogram(
        col, backend, hist.HistogramType.L1_CONTRIBUTIONS)


def _compute_linf_contributions_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of per partition privacy id contributions.

    This histogram contains: the number of (privacy id, partition_key)-pairs
    which have 1 row in the datasets, 2 rows etc.

    Args:
        col: collection with a pre-aggregated dataset, each element is
        (partition_key, (count, sum, n_partitions, n_contributions)).
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed histograms.Histogram.
    """
    linf = backend.map_tuple(
        col,
        lambda _, x: x[0],  # x is (count, sum, n_partitions, n_contributions)
        "Extract count per partition contribution")
    # linf: (int,) where each element is the count of elements the
    # corresponding privacy_id contributes to the partition.
    return _compute_frequency_histogram(linf, backend,
                                        hist.HistogramType.LINF_CONTRIBUTIONS)


def _compute_partition_count_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes histogram of counts per partition.

    This histogram contains: the number of partitions with total count of
    contributions = 1, 2 etc.

    Args:
        col: collection with a pre-aggregated dataset, each element is
        (partition_key, (count, sum, n_partitions, n_contributions)).
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains the computed histograms.Histogram.
    """

    col = backend.map_values(
        col,
        lambda x: x[0],  # x is (count, sum, n_partitions, n_contributions)
        "Extract partition key and count of privacy ID contributions")
    # col: (pk, int)
    col = backend.sum_per_key(col, "Sum per partition")
    # col: (pk, int), where each element is the total count per partition.
    col = backend.values(col, "Drop partition keys")
    # col: (int,)
    return _compute_frequency_histogram(col, backend,
                                        hist.HistogramType.COUNT_PER_PARTITION)


def _compute_partition_privacy_id_count_histogram_on_preaggregated_data(
        col, backend: pipeline_backend.PipelineBackend):
    """Computes a histogram of privacy id counts per partition.

    This histogram contains: the number of partitions with privacy_id_count=1,
    with privacy_id_count=2 etc.

    Args:
        col:collection with a pre-aggregated dataset, each element is
          (partition_key, (count, sum, n_partitions, n_contributions)).
        backend: PipelineBackend to run operations on the collection.

    Returns:
      1 element collection, which contains the computed histograms.Histogram.
  """

    col = backend.keys(col, "Extract partition keys")
    # col: (pk)
    col = backend.count_per_element(col, "Count privacy IDs per partition key")
    # col: (pk, n)
    col = backend.values(col, "Drop partition keys")

    return _compute_frequency_histogram(
        col, backend, hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION)


def compute_dataset_histograms_on_preaggregated_data(
        col, data_extractors: pipeline_dp.PreAggregateExtractors,
        backend: pipeline_backend.PipelineBackend):
    """Computes dataset histograms on pre-aggregated dataset.

    Args:
        col: collection with a pre-aggregated dataset, each element is
          (partition_key, (count, sum, n_partitions, n_contributions)).
        backend: PipelineBackend to run operations on the collection.

    Returns:
        1 element collection, which contains a histograms.DatasetHistograms object.
  """

    col = backend.map(
        col, lambda row: (data_extractors.partition_extractor(row),
                          data_extractors.preaggregate_extractor(row)),
        "Extract (partition_key, preaggregate_data))")
    # col: (partition_key, (count, sum, n_partitions, n_contributions))

    col = backend.to_multi_transformable_collection(col)
    # col: (partition_key, (count, sum, n_partitions, n_contributions))

    # Compute histograms.
    l0_contributions_histogram = _compute_l0_contributions_histogram_on_preaggregated_data(
        col, backend)
    l1_contributions_histogram = _compute_l1_contributions_histogram_on_preaggregated_data(
        col, backend)
    linf_contributions_histogram = _compute_linf_contributions_histogram_on_preaggregated_data(
        col, backend)
    partition_count_histogram = _compute_partition_count_histogram_on_preaggregated_data(
        col, backend)
    partition_privacy_id_count_histogram = _compute_partition_privacy_id_count_histogram_on_preaggregated_data(
        col, backend)

    # Combine histograms to histograms.DatasetHistograms.
    return _to_dataset_histograms([
        l0_contributions_histogram, l1_contributions_histogram,
        linf_contributions_histogram, partition_count_histogram,
        partition_privacy_id_count_histogram
    ], backend)
