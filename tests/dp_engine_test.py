"""DPEngine Test"""

import unittest

from pipeline_dp.aggregate_params import AggregateParams, Metrics
from pipeline_dp.dp_engine import DPEngine

class DPEngineTest(unittest.TestCase):
  def test_aggregate_none(self):
    self.assertIsNone(DPEngine(None, None).aggregate(None, None, None))

  def test_aggregate_report(self):
    params1 = AggregateParams(
      max_partitions_contributed=3,
      max_contributions_per_partition=2,
      low=1,
      high=5,
      metrics=[Metrics.PRIVACY_ID_COUNT, Metrics.COUNT, Metrics.MEAN],
    )
    params2 = AggregateParams(
      max_partitions_contributed=1,
      max_contributions_per_partition=3,
      low=2,
      high=10,
      metrics=[Metrics.VAR, Metrics.SUM, Metrics.MEAN],
      public_partitions = list(range(1,40)),
    )
    engine = DPEngine(None, None)
    engine.aggregate(None, params1, None)
    engine.aggregate(None, params2, None)
    self.assertEqual(len(engine._report_generators), 2)  # pylint: disable=protected-access
    self.assertIn("['p', 'c', 'm']", engine._report_generators[0].report())  # pylint: disable=protected-access
    self.assertIn("['v', 's', 'm']", engine._report_generators[1].report())  # pylint: disable=protected-access

if __name__ == '__main__':
  unittest.main()
