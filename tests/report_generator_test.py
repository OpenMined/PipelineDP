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
"""Report Generator Test"""

import unittest

import pipeline_dp
from pipeline_dp.aggregate_params import AggregateParams, Metrics
from pipeline_dp.report_generator import ReportGenerator


class ReportGeneratorTest(unittest.TestCase):

    def test_report_empty(self):
        self.assertEqual("", ReportGenerator(None, "test_method").report())

    def test_report_params(self):
        expected_report = (
            "DPEngine method: test_method\n"
            "AggregateParams:\n"
            " metrics=['privacy_id_count', 'count', 'mean', 'sum']\n"
            " noise_kind=gaussian\n"
            " budget_weight=1\n"
            " Contribution bounding:\n"
            "  max_partitions_contributed=2\n"
            "  max_contributions_per_partition=1\n"
            "  min_value=1\n"
            "  max_value=5\n"
            "Computation graph:\n"
            " 1. Stage1 \n"
            " 2. Stage2")
        params = AggregateParams(noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                 max_partitions_contributed=2,
                                 max_contributions_per_partition=1,
                                 min_value=1,
                                 max_value=5,
                                 metrics=[
                                     Metrics.PRIVACY_ID_COUNT, Metrics.COUNT,
                                     Metrics.MEAN, Metrics.SUM
                                 ])
        report_generator = ReportGenerator(params, "test_method")
        report_generator.add_stage("Stage1 ")  # add string
        report_generator.add_stage(lambda: "Stage2")  # add lambda returning str
        self.assertEqual(expected_report, report_generator.report())


if __name__ == "__main__":
    unittest.main()
