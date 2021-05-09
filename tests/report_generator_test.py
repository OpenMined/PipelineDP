"""Report Generator Test"""

import unittest

from pipeline_dp.report_generator import ReportGenerator
from pipeline_dp.dp_engine import AggregateParams, Metrics

class ReportGeneratorTest(unittest.TestCase):
  def test_report_empty(self):
    ReportGenerator(None).report()  # no errors

  def test_report_params(self):
    params = AggregateParams(
      max_partitions_contributed=2,
      max_contributions_per_partition=1,
      low=1,
      high=5,
      metrics=[Metrics.PRIVACY_ID_COUNT,Metrics.COUNT,Metrics.MEAN],
    )
    ReportGenerator(params).report() # no errors

  def test_report_aggregate(self):
    params1 = AggregateParams(
      max_partitions_contributed=3,
      max_contributions_per_partition=2,
      low=1,
      high=5,
      metrics=[Metrics.PRIVACY_ID_COUNT,Metrics.COUNT, Metrics.MEAN],
    )
    params2 = AggregateParams(
      max_partitions_contributed=1,
      max_contributions_per_partition=3,
      low=2,
      high=10,
      metrics=[Metrics.VAR,Metrics.SUM,Metrics.MEAN],
      public_partitions = list(range(1,40)),
    )
    engine_stub = DPEngineStub()
    engine_stub.aggregate(params1)
    engine_stub.aggregate(params2)
    for _, report_aggregate in enumerate(engine_stub.report_generators):
      print(report_aggregate.report()) # no errors

class DPEngineStub():
  """Stub DPEngine for testing."""
  def __init__(self):
    self.report_generators = []

  def _add_report_stage(self, text):
    self.report_generators[-1].add_stage(text)

  def aggregate(self, params):
    self.report_generators.append(ReportGenerator(params))
    self._add_report_stage(f"Clip values to {params.low, params.high}")
    self._add_report_stage(
      f"Per-partition contribution: randomly selected not "
      f"more than {params.max_partitions_contributed} contributions")
    self._add_report_stage(
      f"Cross partition contribution bonding: randomly selected not more than "
      f"{params.max_contributions_per_partition} partitions per user")
    if params.public_partitions is None:
      self._add_report_stage("Partitions selection: using thresholding")
    else:
      self._add_report_stage(
        "Partitions selection: using provided public partition")
    noise_beta = [16.0,16.0,80.0,]
    noise_std = [22.62,22.62,113,]
    self._add_report_stage(
      f"Adding laplace random noise with scale={noise_beta} " \
      f"(std={noise_std}) to ({[m.value[0] for m in params.metrics]})" \
      f" per partition.")

if __name__ == "__main__":
  unittest.main()
