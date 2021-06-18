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
        result = col

        # If no public partitions were specified, return aggregation results
        # directly.
        if params.public_partitions is None:
            return result
        else:
            return self._drop_not_public_partitions(result,
                                                    params.public_partitions,
                                                    data_extractors)

    def _drop_not_public_partitions(self, col, public_partitions,
                                    data_extractors):
        return self._ops.filter_by_key(col, public_partitions, data_extractors,
                                       "Filtering out non-public partitions")

    def _bound_contributions(self, col, max_partitions_contributed: int,
                             max_contributions_per_partition: int,
                             aggregator_fn):
        """
    Bounds the contribution by privacy_id in and cross partitions.
    Args:
      col: collection, with types of each element: (privacy_id,
        partition_key, value).
      max_partitions_contributed: maximum number of partitions that one
        privacy id can contribute to.
      max_contributions_per_partition: maximum number of records that one
        privacy id can contribute to one partition.
      aggregator_fn: function that takes a list of values and returns an
        aggregator object which handles all aggregation logic.

    return: collection with elements ((privacy_id, partition_key),
          aggregator).
    """
        # per partition-contribution bounding with bounding of each contribution
        col = self._ops.map_tuple(
            col, lambda pid, pk, v: ((pid, pk), v),
            "Rekey to ( (privacy_id, partition_key), value))")
        col = self._ops.sample_fixed_per_key(
            col, max_contributions_per_partition,
            "Sample per (privacy_id, partition_key)")
        # ((privacy_id, partition_key), [value])
        col = self._ops.map_values(
            col, aggregator_fn,
            "Apply aggregate_fn after per partition bounding")
        # ((privacy_id, partition_key), aggregator)

        # Cross partition bounding
        col = self._ops.map_tuple(
            col, lambda pid_pk, v: (pid_pk[0], (pid_pk[1], v)),
            "Rekey to (privacy_id, (partition_key, "
            "aggregator))")
        col = self._ops.sample_fixed_per_key(col, max_partitions_contributed,
                                             "Sample per privacy_id")

        # (privacy_id, [(partition_key, aggregator)])

        def unnest_cross_partition_bound_sampled_per_key(pid_pk_v):
            pid, pk_values = pid_pk_v
            return (((pid, pk), v) for (pk, v) in pk_values)

        return self._ops.flat_map(col,
                                  unnest_cross_partition_bound_sampled_per_key,
                                  "Unnest")
