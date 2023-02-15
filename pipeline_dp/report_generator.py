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
"""Explain DP computation reports.

An Explain Computation report contains a human readable description of a DP
aggregation performed by PipelineDP. It includes
1. The input parameters (i.e. pipeline_dp.AggregateParams)
2. The main stages the computation graphs.

Example of the report

    DPEngine method: aggregate
    AggregateParams:
     metrics=['SUM']
     noise_kind=laplace
     budget_weight=1
     Contribution bounding:
      max_partitions_contributed=2
      max_contributions_per_partition=10
      min_value=1
      max_value=5
     Partition selection: private partitions
    Computation graph:
     1. Per-partition contribution bounding: for each privacy_id and each partition, randomly select max(actual_contributions_per_partition, 1) contributions.
     2. Cross-partition contribution bounding: for each privacy_id randomly select max(actual_partition_contributed, 2) partitions
     3. Private Partition selection: using Truncated Geometric method with (eps=0.5, delta=1e-06)
     4. Computed sum with (eps=0.5 delta=0)
"""

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

    def add_stage(self, stage_description: Union[Callable, str]) -> None:
        """Add a stage description to the report.

        Args:
            stage_description: description of the stage. Note that it might be
            a Callable that returns str. Support Callable is needed to support
            cases when the description contains information which is not yet
            available during the pipeline construction, e.g. the budget.
        """
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


class ExplainComputationReport:
    """Container explain computation report for 1 DP aggregation."""

    def __init__(self):
        self._report_generator = None

    def _set_report_generator(self, report_generator: ReportGenerator):
        self._report_generator = report_generator

    def text(self) -> str:
        """Returns the text of the report.

        Raises:
            ValueError when this function is called before
              BudgetAccountant.compute_budget().
        """
        if self._report_generator is None:
            raise ValueError("The report_generator is not set.\nWas this object"
                             " passed as an argument to DP aggregation method?")
        try:
            return self._report_generator.report()
        except e:
            raise ValueError("Explain computation report failed to be generated"
                             ".\nWas BudgetAccountant.compute_budget() called?")
