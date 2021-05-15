"""DP aggregations."""

from typing import Callable

from dataclasses import dataclass
from pipeline_dp.aggregate_params import AggregateParams
from pipeline_dp.budget_accounting import BudgetAccountant
from pipeline_dp.pipeline_operations import PipelineOperations
from pipeline_dp.report_generator import ReportGenerator

@dataclass
class DataExtractors:
  """Data extractors

  A set of functions that, given an input, return the privacy id, partition key,
  and value.
  """

  privacy_id_extractor: Callable = None
  partition_extractor: Callable = None
  value_extractor: Callable = None


class DPEngine:
  """Performs DP aggregations."""

  def __init__(self, budget_accountant: BudgetAccountant,
               ops: PipelineOperations):
    self._budget_accountant = budget_accountant
    self._ops = ops
    self._report_generators = []

  def _add_report_stage(self, text):
    self._report_generators[-1].add_stage(text)

  def aggregate(self, col, params: AggregateParams,
                data_extractors: DataExtractors):
    """Computes DP aggregation metrics

    Args:
      col: collection with elements of the same type.
      params: specifies which metrics to compute and computation parameters.
      data_extractors: functions that extract needed pieces of information from
        elements of 'col'
    """
    if params is None:
      return None
    self._report_generators.append(ReportGenerator(params))
    self._add_report_stage(f"Clip values to {params.low, params.high}")
    self._add_report_stage(
      f"Per-partition contribution: randomly selected not "
      f"more than {params.max_partitions_contributed} contributions")
    self._add_report_stage(
      f"Cross partition contribution bounding: randomly selected not"
      f"more than {params.max_contributions_per_partition} partitions per"
      " user")
    if params.public_partitions is None:
      self._add_report_stage("Partitions selection: using thresholding")
    else:
      self._add_report_stage(
        "Partitions selection: using provided public partition")
    # TODO: implement aggregate().
    # It returns input for now, just to ensure that the an example works.
    return col
