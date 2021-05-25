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
                data_extractors: DataExtractors):  # pylint: disable=unused-argument
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
    # TODO: implement aggregate().
    # It returns input for now, just to ensure that the an example works.
    return col

  def _bound_cross_partition_contributions(self, col,
                                           max_partitions_contributed: int,
                                           max_contributions_per_partition: int,
                                           aggregator_fn):
    """
    Bounds the contribution by privacy_id in and cross partitions
    Args:
      col: collection, with types of each element: (privacy_id,
        partition_key, value)
      max_partitions_contributed: maximum number of partitions that one
        privacy id can contribute to
      max_contributions_per_partition: maximum number of records that one
        privacy id can contribute to one partition
      aggregator_fn: function that takes a list of values and returns an
        aggregator object which handles all aggregation logic.

    return: collection with elements ((privacy_id, partition_key),
          aggregator)
    """
    # per partition-contribution bounding with bounding of each contribution
    col = self._ops.map_tuple(col, lambda pid, pk, v: ((pid, pk), v),
                              "To (privacy_id, partition_key), value))")
    col = self._ops.sample_fixed_per_key(col,
                                         max_contributions_per_partition,
                                         "Sample per (privacy_id, partition_key)")
    # ((privacy_id, partition_key), [value])
    col = self._ops.map(col, lambda pid_pk: (pid_pk[0], aggregator_fn(
      pid_pk[1])), "Apply aggregate_fn after per partition bounding")
    # ((privacy_id, partition_key), aggregator)

    # Cross partition bounding
    col = self._ops.map_tuple(col, lambda pid_pk, v: (pid_pk[0],
                                                      (pid_pk[1], v)),
                              "To (privacy_id, (partition_key, aggregator))")
    col = self._ops.sample_fixed_per_key(col, max_partitions_contributed,
                                         "Sample per privacy_id")
    # (privacy_id, [(partition_key, aggregator)])
    return self._ops.flat_map(col, lambda pid: [((pid[0], pk_v[0]), pk_v[1])
                                                for pk_v in pid[1]],
                              "Unnest")

