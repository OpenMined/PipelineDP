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
"""Functions for computing count histograms."""

# This file contains histograms which are useful for analysis of DP SUM
# aggregation utility.
# The general structure of these histogram is the following:
# The input is a collection of positive integer values X = (k_0, ... k_n).
# The output histogram contain bins, each bin contain the count of elements
# which are in [lower, upper).
# [lower, upper) are generated in the following (logarithmic) manner:
# [1,2), ... [999, 1000)
# [1000, 1010), ... [9990, 10000)
# [10000, 10100),...
# etc

import operator
from typing import Tuple

from pipeline_dp import pipeline_backend
from pipeline_dp.dataset_histograms import histograms as hist


def _to_bin_lower_upper_logarithmic(value: int) -> Tuple[int, int]:
    """Finds the lower and upper bounds of the histogram bin which contains
  the given integer.

  Keep algorithm in sync with
  private_contribution_bounds._generate_possible_contribution_bounds.
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
    lower = value // round_base * round_base
    bin_size = round_base if value != bound else round_base * 10
    return lower, lower + bin_size


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
    return _compute_frequency_histogram_helper(col, backend, name)


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
    return _compute_frequency_histogram_helper(col, backend, name)


def _compute_frequency_histogram_helper(
        col, backend: pipeline_backend.PipelineBackend,
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
        bin_lower, bin_upper = _to_bin_lower_upper_logarithmic(value)
        return bin_lower, hist.FrequencyBin(lower=bin_lower,
                                            upper=bin_upper,
                                            count=frequency,
                                            sum=frequency * value,
                                            max=value,
                                            min=value)

    col = backend.map_tuple(col, _map_to_frequency_bin, "To FrequencyBin")
    # (lower_bin_value, hist.FrequencyBin)
    return _convert_frequency_bins_into_histogram(col, backend, name)


def _convert_frequency_bins_into_histogram(
        col, backend: pipeline_backend.PipelineBackend, name):
    """Converts (lower_bin_value, hist.FrequencyBin) into histogram.

  The input collection is not expected to have frequency bins reduced per
  lower values.
  """
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


def _to_dataset_histograms(histogram_list,
                           backend: pipeline_backend.PipelineBackend):
    """Combines histogram_list to hist.DatasetHistograms."""
    histograms = backend.flatten(histogram_list, "Histograms to one collection")
    # histograms: 5 elements collection with elements ContributionHistogram

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
