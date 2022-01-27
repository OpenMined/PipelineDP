"""Report Generator Test"""

import unittest

import pipeline_dp
from pipeline_dp.aggregate_params import AggregateParams, Metrics
from pipeline_dp.report_generator import ReportGenerator


class ReportGeneratorTest(unittest.TestCase):

    def test_report_empty(self):
        self.assertEqual("", ReportGenerator(None).report())

    def test_report_params(self):
        test_report = ("Differentially private: Computing <Metrics: "
                       "['privacy_id_count', 'count', 'mean', 'sum']>"
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
                                     Metrics.MEAN, Metrics.SUM
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
