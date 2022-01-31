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


class ReportGenerator:
    """Generates a report based on the metrics and stages in the pipeline.

  Each ReportGenerator corresponds to one aggregation which contains an
  ordered set of stages. It collects information about the DP aggregation
  and generates a report.
  """

    def __init__(self, params):
        """Initialize the ReportGenerator."""
        self._params_str = None
        if params:
            self._params_str = str(params)
        self._stages = []

    def add_stage(self, text: str): #not text
        """Add a stage description to the report."""
        self._stages.append(text)

    def add_stages(self, lines):
        """Add a stage description to the report."""
        self._stages.extend(lines)

    def report(self) -> str:
        """Constructs a report based on stages and metrics."""
        if not self._params_str:
            return ""
        title = f"Computing <{self._params_str}>"
        result = [f"Differentially private: {title}"]
        for i, stage_str in enumerate(self._stages):
            if hasattr(stage_str, "__call__"):
                result.append(f"{i+1}. {stage_str()}")
            else:
                result.append(f"{i+1}. {stage_str}")
        return "\n".join(result)
