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
        self.assertEqual("", ReportGenerator(None).report())

    def test_report_params(self):
        test_report = (
            "Differentially private: Computing <Metrics: "
            "['privacy_id_count', 'count', 'mean', 'sum', 'variance']>"
            "\n1. Eat between (1, 5) snacks"
            "\n2. Eat a maximum of snack varieties total: 2"
            "\n3. Eat a maximum of a single snack variety: 1")
        params = AggregateParams(noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                 max_partitions_contributed=2,
                                 max_contributions_per_partition=1,
                                 min_value=1,
                                 max_value=5,
                                 metrics=[
                                     Metrics.PRIVACY_ID_COUNT, Metrics.COUNT,
                                     Metrics.MEAN, Metrics.SUM, Metrics.VAR
                                 ])
        generated_report = aggregate_stub(params)
        self.assertIn(test_report, generated_report)


def aggregate_stub(params: AggregateParams) -> str:
    report_generator = ReportGenerator(params)
    report_generator.add_stage(
        f"Eat between {params.min_value, params.max_value} snacks")
    report_generator.add_stage(("Eat a maximum of snack varieties total: "
                                f"{params.max_partitions_contributed}"))
    report_generator.add_stage(("Eat a maximum of a single snack variety: "
                                f"{params.max_contributions_per_partition}"))
    return report_generator.report()


if __name__ == "__main__":
    unittest.main()
