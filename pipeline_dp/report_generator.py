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
"""Generator for explaining DP computation reports."""

from pipeline_dp import aggregate_params as agg

from typing import Optional, Union, Callable


class ReportGenerator:
    """Generates a report based on the metrics and stages in the pipeline.

  Each ReportGenerator corresponds to one aggregation which contains an
  ordered set of stages. It collects information about the DP aggregation
  and generates a report.
  """

    def __init__(self,
                 params,
                 method_name: str,
                 is_public_partition: Optional[bool] = None):
        """Initialize the ReportGenerator."""
        self._params_str = None
        if params:
            self._params_str = agg.parameters_to_readable_string(
                params, is_public_partition)
        self._method_name = method_name
        self._stages = []

    def add_stage(self, stage_description: Union[Callable, str]):
        """Add a stage description to the report."""
        self._stages.append(stage_description)

    def report(self) -> str:
        """Constructs a report based on stages and metrics."""
        if not self._params_str:
            return ""
        result = [f"DPEngine method: {self._method_name}"]
        result.append(self._params_str)
        result.append("Computation graph:")
        for i, stage_str in enumerate(self._stages):
            if hasattr(stage_str, "__call__"):
                result.append(f" {i+1}. {stage_str()}")
            else:
                result.append(f" {i+1}. {stage_str}")
        return "\n".join(result)
