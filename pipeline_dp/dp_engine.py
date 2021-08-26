"""DP aggregations."""

from functools import partial
from typing import Any, Callable, Tuple

from dataclasses import dataclass
from pipeline_dp.aggregate_params import AggregateParams
from pipeline_dp.budget_accounting import BudgetAccountant, Budget
from pipeline_dp.pipeline_operations import PipelineOperations
from pipeline_dp.report_generator import ReportGenerator
from pipeline_dp.accumulator import Accumulator
from pipeline_dp.accumulator import AccumulatorFactory

from pydp.algorithms.partition_selection import create_truncated_geometric_partition_strategy


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

        accumulator_factory = AccumulatorFactory(
            params=params, budget_accountant=self._budget_accountant)
        accumulator_factory.initialize()
        aggregator_fn = accumulator_factory.create

        # extract the columns
        col = self._ops.map_tuple(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row),
                              data_extractors.value_extractor(row)),
            "Extract (privacy_id, partition_key, value))")
        # col : (privacy_id, partition_key, value)
        col = self._bound_contributions(col, params.max_partitions_contributed,
                                        params.max_contributions_per_partition,
                                        aggregator_fn)
        # col : ((privacy_id, partition_key), accumulator)
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

    def _select_private_partitions(self, col, max_partitions_contributed: int):
        """Selects and publishes private partitions.

        Args:
            col: collection, with types for each element: 
                (partition_key, Accumulator)
            max_partitions_contributed: maximum amount of partitions that one privacy unit
                might contribute.
        
        Returns:
            collection of elements (partition_key, accumulator)
        """
        budget = self._budget_accountant.request_budget(weight=1,
                                                        use_eps=True,
                                                        use_delta=True)

        def filter_fn(captures: Tuple[Budget, int],
                      row: Tuple[Any, Accumulator]) -> bool:
            """Lazily creates a partition selection strategy and uses it to determine which 
            partitions to keep."""
            budget, max_partitions = captures
            accumulator = row[1]
            partition_selection_strategy = create_truncated_geometric_partition_strategy(
                budget.eps, budget.delta, max_partitions)
            return partition_selection_strategy.should_keep(
                accumulator.privacy_id_count)

        # make filter_fn serializable
        filter_fn = partial(filter_fn, (budget, max_partitions_contributed))
        return self._ops.filter(col, filter_fn)
