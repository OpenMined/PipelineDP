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
"""DPEngine and ContributionBounder for utility analysis."""
import collections

import pipeline_dp
from pipeline_dp import DPEngine
from pipeline_dp import DataExtractors
from pipeline_dp import NoiseKind
from pipeline_dp import contribution_bounders
from pipeline_dp.contribution_bounders import ContributionBounder
from dataclasses import dataclass


class UtilityAnalysisDPEngine(DPEngine):

    def __init__(self, budget_accountant: 'BudgetAccountant',
                 backend: 'PipelineBackend'):
        super().__init__(budget_accountant, backend)

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: DataExtractors,
                  public_partitions=None):
        return super().aggregate(col, params, data_extractors,
                                 public_partitions)

    def _aggregate(self, col, params: pipeline_dp.AggregateParams,
                   data_extractors: DataExtractors, public_partitions):
        assert params.custom_combiners is None, "Cumtom combiners are not supported"
        assert params.metrics == [
            pipeline_dp.Metrics.COUNT
        ], f"Supported only count metrics, metrics={params.metrics}"
        assert public_partitions is not None, "Only public partitions supported"
        assert params.contribution_bounds_already_enforced is False, "Utility Analysis when contribution bounds are already enforced is not supported"

        # TODO: Implement custom combiner
        combiner = self._create_compound_combiner(params)
        # Extract the columns.
        col = self._backend.map(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row),
                              data_extractors.value_extractor(row)),
            "Extract (privacy_id, partition_key, value))")
        # col : (privacy_id, partition_key, value)
        contribution_bounder = self._create_contribution_bounder(params)
        col = contribution_bounder.bound_contributions(
            col, params, self._backend, self._current_report_generator,
            combiner.create_accumulator)
        # col : ((privacy_id, partition_key), accumulator)

        col = self._backend.map_tuple(col, lambda pid_pk, v: (pid_pk[1], v),
                                      "Drop privacy id")
        # col : (partition_key, accumulator)

        # Compute DP metrics.
        self._add_report_stages(combiner.explain_computation())
        col = self._backend.map_values(col, combiner.compute_metrics,
                                       "Compute DP` metrics")

        return col

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams
    ) -> contribution_bounders.ContributionBounder:
        """Creates Utility Analysis ContributionBounder."""
        return UtilityAnalysisSamplingCrossAndPerPartitionContributionBounder()


class UtilityAnalysisSamplingCrossAndPerPartitionContributionBounder(
        ContributionBounder):
    """'Bounds' the contribution by privacy_id per and cross partitions.

  Because this is for Utility Analysis, it doesn't actually ensure that
  contribution bounds are enforced. Instead, it keeps track of probabilities
  different data points are kept under a per-partition and cross-partition
  contribution bounder.

  Only works for count at the moment.
  """

    def bound_contributions(self, col, params, backend, report_generator,
                            aggregate_fn):
        """See docstrings for this class and the base class."""
        max_partitions_contributed = params.max_partitions_contributed
        max_contributions_per_partition = params.max_contributions_per_partition
        col = backend.map_tuple(
            col, lambda pid, pk, v: (pid, (pk, v)),
            "Rekey to ((privacy_id), (partition_key, value))")

        # Convert the per privacy id list into a dict with key as partition_key
        # and values as the list of input values.
        def collect_values_per_partition_key_per_privacy_id(input_list):
            d = collections.defaultdict(list)
            for key, value in input_list:
                d[key].append(value)
            return d

        col = backend.map_values(
            col, collect_values_per_partition_key_per_privacy_id,
            "Group per (privacy_id, partition_key)")

        # Rekey by (privacy_id, partition_key) and unnest values along with the
        # number of partitions contributed per privacy_id.
        def rekey_per_privacy_id_per_partition_key_and_unnest(pid_pk_dict):
            privacy_id, partition_dict = pid_pk_dict
            num_partitions_contributed = len(partition_dict)
            for partition_key, values in partition_dict.items():
                yield (privacy_id, partition_key), (values,
                                                    num_partitions_contributed)

        # Unnest the list per privacy id.
        col = backend.flat_map(
            col, rekey_per_privacy_id_per_partition_key_and_unnest, "Unnest")

        return backend.map_values(
            col, aggregate_fn,
            "Apply aggregate_fn with num_partitions_contributed as input")


@dataclass
class ErrorPrimitives:
    raw_count: int
    # expected value of deviation from raw_count due to per-partition bounding
    # (always nonpositive)
    per_partition_bounding_error: int
    # expected value of deviation from raw_count due to cross-partition bounding
    # (always nonpositive)
    cross_partition_bounding_error: float
    # variance of deviation from raw_count due to cross-partition bounding
    cross_partition_bounding_variance: float
    noise_type: NoiseKind
    noise_std: float
