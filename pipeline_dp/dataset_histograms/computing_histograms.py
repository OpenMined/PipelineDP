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

from typing import Iterable, List, Union
import pipeline_dp
from pipeline_dp import pipeline_backend
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import count_histogram_computation
from pipeline_dp.dataset_histograms import sum_histogram_computation


def _list_to_contribution_histograms(
    histograms: List[Union[hist.Histogram, List[hist.Histogram]]],
) -> hist.DatasetHistograms:
    """Packs histograms from a list to ContributionHistograms."""
    l0_contributions = l1_contributions = None
    linf_contributions = linf_sum_contributions = None
    linf_sum_contributions_log = None
    count_per_partition = privacy_id_per_partition_count = None
    sum_per_partition_histogram = sum_per_partition_histogram_log = None
    for histogram in histograms:
        if isinstance(histogram, Iterable):
            if not histogram:
                # no histograms were computed, this can happen if the dataset is
                # empty
                continue
            histogram_type = histogram[0].name
        else:
            histogram_type = histogram.name

        if histogram_type == hist.HistogramType.L0_CONTRIBUTIONS:
            l0_contributions = histogram
        elif histogram_type == hist.HistogramType.L1_CONTRIBUTIONS:
            l1_contributions = histogram
        elif histogram_type == hist.HistogramType.LINF_CONTRIBUTIONS:
            linf_contributions = histogram
        elif histogram_type == hist.HistogramType.LINF_SUM_CONTRIBUTIONS:
            linf_sum_contributions = histogram
        elif histogram_type == hist.HistogramType.LINF_SUM_CONTRIBUTIONS_LOG:
            linf_sum_contributions_log = histogram
        elif histogram_type == hist.HistogramType.COUNT_PER_PARTITION:
            count_per_partition = histogram
        elif histogram_type == hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION:
            privacy_id_per_partition_count = histogram
        elif histogram_type == hist.HistogramType.SUM_PER_PARTITION:
            sum_per_partition_histogram = histogram
        elif histogram_type == hist.HistogramType.SUM_PER_PARTITION_LOG:
            sum_per_partition_histogram_log = histogram

    return hist.DatasetHistograms(
        l0_contributions,
        l1_contributions,
        linf_contributions,
        linf_sum_contributions,
        linf_sum_contributions_log,
        count_per_partition,
        privacy_id_per_partition_count,
        sum_per_partition_histogram,
        sum_per_partition_histogram_log,
    )


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


def compute_dataset_histograms(col, data_extractors: pipeline_dp.DataExtractors,
                               backend: pipeline_backend.PipelineBackend):
    """Computes dataset histograms.

    Args:
        col: collection with elements of the same type.
        backend: PipelineBackend to run operations on the collection.
    Returns:
        1 element collection, which contains a 'histograms.DatasetHistograms'
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
    l0_contributions_histogram = count_histogram_computation._compute_l0_contributions_histogram(
        col_distinct, backend)
    l1_contributions_histogram = count_histogram_computation._compute_l1_contributions_histogram(
        col, backend)
    linf_contributions_histogram = count_histogram_computation._compute_linf_contributions_histogram(
        col, backend)
    linf_sum_contributions_histograms = sum_histogram_computation._compute_linf_sum_contributions_histogram(
        col_with_values, backend)
    partition_count_histogram = count_histogram_computation._compute_partition_count_histogram(
        col, backend)
    partition_privacy_id_count_histogram = count_histogram_computation._compute_partition_privacy_id_count_histogram(
        col_distinct, backend)
    partition_sum_histograms = sum_histogram_computation._compute_partition_sum_histogram(
        col_with_values, backend)
    # all histograms are 1 element collections which contains ContributionHistogram

    # Combine histograms to histograms.DatasetHistograms.
    return _to_dataset_histograms([
        l0_contributions_histogram, l1_contributions_histogram,
        linf_sum_contributions_histograms, linf_contributions_histogram,
        partition_count_histogram, partition_privacy_id_count_histogram,
        partition_sum_histograms
    ], backend)


def compute_dataset_histograms_on_preaggregated_data(
        col, data_extractors: pipeline_dp.PreAggregateExtractors,
        backend: pipeline_backend.PipelineBackend):
    """Computes dataset histograms on pre-aggregated dataset.

    Args:
        col: collection with a pre-aggregated dataset, each element is
          (partition_key, (count, sum, n_partitions, n_contributions)).
        backend: PipelineBackend to run operations on the collection.

    Returns:
        1 element collection, which contains a 'histograms.DatasetHistograms' object.
  """

    col = backend.map(
        col, lambda row: (data_extractors.partition_extractor(row),
                          data_extractors.preaggregate_extractor(row)),
        "Extract (partition_key, preaggregate_data))")
    # col: (partition_key, (count, sum, n_partitions, n_contributions))

    col = backend.to_multi_transformable_collection(col)
    # col: (partition_key, (count, sum, n_partitions, n_contributions))

    # Compute histograms.
    l0_contributions_histogram = count_histogram_computation._compute_l0_contributions_histogram_on_preaggregated_data(
        col, backend)
    l1_contributions_histogram = count_histogram_computation._compute_l1_contributions_histogram_on_preaggregated_data(
        col, backend)
    linf_contributions_histogram = count_histogram_computation._compute_linf_contributions_histogram_on_preaggregated_data(
        col, backend)
    linf_sum_contributions_histograms = sum_histogram_computation._compute_linf_sum_contributions_histogram_on_preaggregated_data(
        col, backend)
    partition_count_histogram = count_histogram_computation._compute_partition_count_histogram_on_preaggregated_data(
        col, backend)
    partition_privacy_id_count_histogram = count_histogram_computation._compute_partition_privacy_id_count_histogram_on_preaggregated_data(
        col, backend)
    partition_sum_histograms = sum_histogram_computation._compute_partition_sum_histogram_on_preaggregated_data(
        col, backend)

    # Combine histograms to histograms.DatasetHistograms.
    return _to_dataset_histograms([
        l0_contributions_histogram,
        l1_contributions_histogram,
        linf_contributions_histogram,
        linf_sum_contributions_histograms,
        partition_count_histogram,
        partition_sum_histograms,
        partition_privacy_id_count_histogram,
    ], backend)
