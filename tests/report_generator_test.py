"""Report Generator Test"""

import unittest

from pipeline_dp.aggregate_params import AggregateParams, Metrics
from pipeline_dp.report_generator import ReportGenerator

class ReportGeneratorTest(unittest.TestCase):
  def test_report_empty(self):
    self.assertEqual("", ReportGenerator(None).report())

  def test_report_params(self):
    test_report = ("Differential private: Computing metrics: ['p', 'c', 'm']"
      "\n1. Clip values to (1, 5)\n2. Per-partition contribution: 2"
      "\n3. Cross partition contribution bounding: 1")
    params = AggregateParams(
      max_partitions_contributed=2,
      max_contributions_per_partition=1,
      low=1,
      high=5,
      metrics=[Metrics.PRIVACY_ID_COUNT, Metrics.COUNT, Metrics.MEAN],
    )
    generated_report = aggregate_stub(params)
    self.assertIn(test_report, generated_report)

def aggregate_stub(params: AggregateParams) -> str:
  report_generator = ReportGenerator(params)
  report_generator.add_stage(f"Clip values to {params.low, params.high}")
  report_generator.add_stage(("Per-partition contribution: "
    f"{params.max_partitions_contributed}"))
  report_generator.add_stage(("Cross partition contribution bounding: "
    f"{params.max_contributions_per_partition}"))
  return report_generator.report()

if __name__ == "__main__":
  unittest.main()
