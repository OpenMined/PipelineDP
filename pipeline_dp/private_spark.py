from dataclasses import dataclass
from pyspark import RDD
from typing import Callable

import pipeline_dp
from pipeline_dp.budget_accounting import BudgetAccountant


@dataclass
class SumParams:
    """Specifies parameters for sum calculation from PrivateRDD.

    Args:
        max_partitions_contributed: Bounds the number of partitions in which one
            unit of privacy (e.g., a user) can participate.
        max_contributions_per_partition: Bounds the number of times one unit of
            privacy (e.g. a user) can contribute to a partition.
        low: Lower bound on a value contributed by a unit of privacy in a partition.
        high: Upper bound on a value contributed by a unit of privacy in a
            partition.
        public_partitions: A collection of partition keys that will be present in
            the result.
        partition_extractor: A function for partition id extraction from an RDD record.
        value_extractor: A function for extraction of value
            for which the sum will be calculated.
  """

    max_partitions_contributed: int
    max_contributions_per_partition: int
    low: float
    high: float
    budget_weight: float
    public_partitions: list
    partition_extractor: Callable
    value_extractor: Callable


class PrivateRDD:
    """A Spark RDD counterpart in PipelineDP guaranteeing that only anonymized data
    within the specified privacy budget can be extracted from it through its API.
    """

    def __init__(self, rdd, budget_accountant, privacy_id_extractor):
        self.__rdd = rdd
        self.__budget_accountant = budget_accountant
        self.__privacy_id_extractor = privacy_id_extractor

    def sum(self, sum_params: SumParams):
        """Computes sum.

        Args:
            sum_params: parameters for calculation
        """

        ops = pipeline_dp.SparkRDDOperations()
        dp_engine = pipeline_dp.DPEngine(self.__budget_accountant, ops)

        params = pipeline_dp.AggregateParams(
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.max_contributions_per_partition,
            low=sum_params.low,
            high=sum_params.high,
            public_partitions=sum_params.public_partitions
        )

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=sum_params.partition_extractor,
            privacy_id_extractor=self.__privacy_id_extractor,
            value_extractor=sum_params.value_extractor
        )

        dp_result = dp_engine.aggregate(self.__rdd, params, data_extractors)

        return dp_result


def make_private(rdd: RDD, budget_accountant: BudgetAccountant,
                 privacy_id_extractor: Callable):
    """A factory method for PrivateRDD instance creation."""
    prdd = PrivateRDD(rdd, budget_accountant, privacy_id_extractor)
    return prdd
