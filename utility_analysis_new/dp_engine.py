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
"""DPEngine for utility analysis."""

import pipeline_dp
from pipeline_dp import contribution_bounders
import utility_analysis_new.contribution_bounders as utility_contribution_bounders


class UtilityAnalysisEngine(pipeline_dp.DPEngine):
    """Performs utility analysis for DP aggregations."""

    def __init__(self, budget_accountant: 'BudgetAccountant',
                 backend: 'PipelineBackend'):
        super().__init__(budget_accountant, backend)

    def aggregate(self,
                  col,
                  params: pipeline_dp.AggregateParams,
                  data_extractors: pipeline_dp.DataExtractors,
                  public_partitions=None):
        _check_utility_analysis_params(params, public_partitions)
        return super().aggregate(col, params, data_extractors,
                                 public_partitions)

    def _create_contribution_bounder(
        self, params: pipeline_dp.AggregateParams
    ) -> contribution_bounders.ContributionBounder:
        """Creates ContributionBounder for utility analysis."""
        return utility_contribution_bounders.SamplingCrossAndPerPartitionContributionBounder(
        )


def _check_utility_analysis_params(params: pipeline_dp.AggregateParams,
                                   public_partitions=None):
    assert params.custom_combiners is None, "Custom combiners are not supported"
    assert params.metrics == [
        pipeline_dp.Metrics.COUNT
    ], f"Supported only count metrics, metrics={params.metrics}"
    assert public_partitions is not None, "Only public partitions supported"
    assert not params.contribution_bounds_already_enforced, "Utility Analysis when contribution bounds are already enforced is not supported"
