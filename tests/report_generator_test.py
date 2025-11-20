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

from pipeline_dp import aggregate_params as ap
from pipeline_dp import report_generator as rgen


class ReportGeneratorTest(unittest.TestCase):

    def test_report_empty(self):
        self.assertEqual("", rgen.ReportGenerator(None, "test_method").report())

    def test_report_params(self):
        expected_report = (
            "DPEngine method: test_method\n"
            "AggregateParams:\n"
            " metrics=['PRIVACY_ID_COUNT', 'COUNT', 'MEAN', 'SUM']\n"
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
        params = ap.AggregateParams(noise_kind=ap.NoiseKind.GAUSSIAN,
                                    max_partitions_contributed=2,
                                    max_contributions_per_partition=1,
                                    min_value=1,
                                    max_value=5,
                                    metrics=[
                                        ap.Metrics.PRIVACY_ID_COUNT,
                                        ap.Metrics.COUNT, ap.Metrics.MEAN,
                                        ap.Metrics.SUM
                                    ])
        report_generator = rgen.ReportGenerator(params, "test_method")
        report_generator.add_stage("Stage1 ")  # add string
        report_generator.add_stage(lambda: "Stage2")  # add lambda returning str
        self.assertEqual(expected_report, report_generator.report())


class ExplainComputationReportTest(unittest.TestCase):

    def test_report_empty(self):
        report = rgen.ExplainComputationReport()
        with self.assertRaisesRegex(ValueError,
                                    "The report_generator is not set"):
            report.text()

    def test_fail_to_generate(self):
        report = rgen.ExplainComputationReport()
        report_generator = rgen.ReportGenerator(None, "test_method")

        # Simulate that one of the stages of report generation failed.
        def stage_fn():
            raise ValueError("Fail to generate")

        report_generator.add_stage(lambda: stage_fn)

        with self.assertRaisesRegex(ValueError, "report_generator is not set"):
            report.text()

    def test_generate(self):
        report = rgen.ExplainComputationReport()
        params = ap.AggregateParams(noise_kind=ap.NoiseKind.LAPLACE,
                                    metrics=[ap.Metrics.COUNT],
                                    max_partitions_contributed=2,
                                    max_contributions_per_partition=1)
        report_generator = rgen.ReportGenerator(params, "test_method")
        report_generator.add_stage("stage 1")
        report_generator.add_stage("stage 2")
        report._set_report_generator(report_generator)

        text = report.text()
        self.assertTrue("stage 1" in text)
        self.assertTrue("stage 2" in text)


if __name__ == "__main__":
    unittest.main()
