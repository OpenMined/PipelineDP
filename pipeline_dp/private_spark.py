from pyspark import RDD
from typing import Callable

import pipeline_dp
from pipeline_dp import aggregate_params, budget_accounting


class PrivateRDD:
    """ A Spark RDD counterpart.

    PrivateRDD guarantees that only anonymized data
    within the specified privacy budget can be extracted from it through its API.
    """

    def __init__(self, rdd, budget_accountant, privacy_id_extractor):
        self._rdd = rdd
        self._budget_accountant = budget_accountant
        self._privacy_id_extractor = privacy_id_extractor

    def sum(self, sum_params: aggregate_params.SumParams) -> RDD:
        """Computes DP sum.

        Args:
            sum_params: parameters for calculation
        """

        ops = pipeline_dp.SparkRDDOperations()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, ops)

        params = pipeline_dp.AggregateParams(
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.
                max_contributions_per_partition,
            low=sum_params.low,
            high=sum_params.high,
            public_partitions=sum_params.public_partitions)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=sum_params.partition_extractor,
            privacy_id_extractor=self._privacy_id_extractor,
            value_extractor=sum_params.value_extractor)

        dp_result = dp_engine.aggregate(self._rdd, params, data_extractors)

        return dp_result


def make_private(rdd: RDD,
                 budget_accountant: budget_accounting.BudgetAccountant,
                 privacy_id_extractor: Callable) -> PrivateRDD:
    """A factory method for PrivateRDD instance creation."""
    prdd = PrivateRDD(rdd, budget_accountant, privacy_id_extractor)
    return prdd
