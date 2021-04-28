"""DP aggregations."""

from enum import Enum
from typing import Any, Callable, Iterable

from dataclasses import dataclass
from pipeline_dp.budget_accounting import BudgetAccountant
from pipeline_dp.pipeline_operations import PipelineOperations


class NoiseKind(Enum):
  LAPLACE = 'laplace'
  GAUSSIAN = 'gaussian'


class Metrics(Enum):
  COUNT = 'count',
  PRIVACY_ID_COUNT = 'privacy_id_count',
  SUM = 'sum',
  MEAN = 'mean',
  VAR = 'variance'


@dataclass
class AggregateParams:
  """Specifies parameters for function DPEngine.aggregate()

  Args:
    metrics: Metrics to compute.
    max_partitions_contributed: Bounds the number of partitions in which one
      unit of privacy (e.g., a user) can participate.
    max_contributions_per_partition: Bounds the number of times one unit of
      privacy (e.g. a user) can contribute to a partition.
    low: Lower bound on a value contributed by a unit of privacy in a partition.
    high: Upper bound on a value contributed by a unit of privacy in a
      partition.
    public_partitions: a collection of partition keys that will be present in
      the result.
  """

  metrics: Iterable[Metrics]
  max_partitions_contributed: int
  max_contributions_per_partition: int
  low: float = None
  high: float = None
  budget_weight: float = 1
  public_partitions: Any = None


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
