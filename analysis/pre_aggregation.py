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
import utility_analysis_new.contribution_bounders as utility_contribution_bounders


def preaggregate(col,
                 backend: pipeline_dp.PipelineBackend,
                 data_extractors: pipeline_dp.DataExtractors,
                 partitions_sampling_prob: float = 1):
    """Preaggregates data from a collection.

    The output is a collection with elements
      (partition_key, (count, sum, n_partitions)).
    Each element corresponds to each (privacy_id, partition_key) which is
    present in the dataset. count and sum correspond to count and sum of values
    contributed by the privacy_id to the partition_key. n_partitions is the
    number of distinct partitions contributed by the privacy_id.
    If partitions_sampling_prob < 1, the output partitions will be sampled.

    Args:
        col: collection where all elements are of the same type.
        backend: PipelineBackend for performing transformations on collections.
        data_extractors: functions that extract needed pieces of information
          from elements of 'col'.
        partitions_sampling_prob: the probability with which each partition
          will be sampled. It is useful for speed-up computations on the large
          datasets.

    Returns:
        a collection with elements (partition_key, (count, sum, n_partitions)).
    """
    col = backend.map(
        col, lambda row: (data_extractors.privacy_id_extractor(row),
                          data_extractors.partition_extractor(row),
                          data_extractors.value_extractor(row)),
        "Extract (privacy_id, partition_key, value))")
    # col: (privacy_id, partition_key, value):
    bounder = utility_contribution_bounders.SamplingL0LinfContributionBounder(
        partitions_sampling_prob)
    col = bounder.bound_contributions(col,
                                      params=None,
                                      backend=backend,
                                      report_generator=None,
                                      aggregate_fn=lambda x: x)
    # col: ((privacy_id, partition_key), (count, sum, n_partitions)).

    return backend.map(col, lambda row: (row[0][1], row[1]), "Drop privacy id")
    # (partition_key, (count, sum, n_partitions))
