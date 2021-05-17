"""DP aggregations."""

from typing import Callable

from dataclasses import dataclass
from pipeline_dp.aggregate_params import AggregateParams
from pipeline_dp.budget_accounting import BudgetAccountant
from pipeline_dp.pipeline_operations import PipelineOperations

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

  def aggregate(self, col, params: AggregateParams,
                data_extractors: DataExtractors):
    """Computes DP aggregation metrics

    Args:
      col: collection with elements of the same type.
      params: specifies which metrics to compute and computation parameters.
      data_extractors: functions that extract needed pieces of information from
        elements of 'col'
    """

    # TODO: implement aggregate().
    # It returns input for now, just to ensure that the an example works.
    return col
