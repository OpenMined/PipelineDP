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
"""API functions for performing utility analysis."""

import pipeline_dp
from dataclasses import dataclass


@dataclass
class UtilityAnalysisOptions:
    """Options for the utility analysis."""
    eps: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams


def perform_utility_analysis(col,
                             options: UtilityAnalysisOptions,
                             data_extractors: pipeline_dp.DataExtractors,
                             public_partitions=None):
    """Performs utility analysis for DP aggregations.

  Args:
    col: collection where all elements are of the same type.
    options: options for utility analysis.
    data_extractors: functions that extract needed pieces of information
          from elements of 'col'.
    public_partitions: A collection of partition keys that will be present
          in the result.
  """
    raise NotImplementedError(
        "perform_utility_analysis is not implemented yet.")
