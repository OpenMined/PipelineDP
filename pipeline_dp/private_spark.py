from pyspark import RDD
from typing import Callable

import pipeline_dp
from pipeline_dp import aggregate_params, budget_accounting


class PrivateRDD:
    """A Spark RDD counterpart.

    PrivateRDD guarantees that only anonymized data
    within the specified privacy budget can be extracted from it through its API.

    PrivateRDD keeps `privacy_id` for each element
    in order to guarantees correct DP computations.
    """

    def __init__(self, rdd, budget_accountant, privacy_id_extractor=None):
        if privacy_id_extractor:
            self._rdd = rdd.map(lambda x: (privacy_id_extractor(x), x))
        else:
            # It's assumed that rdd is already in format (privacy_id, value)
            self._rdd = rdd
        self._budget_accountant = budget_accountant

    def map(self, fn: Callable) -> 'PrivateRDD':
        """A Spark map equivalent.

        Keeps track of privacy_id for each element.
        The output PrivateRDD has the same BudgetAccountant as self.
        """
        # Assumes that `self._rdd` consists of tuples `(privacy_id, element)`
        # and transforms each `element` according to the supplied function `fn`.
        rdd = self._rdd.mapValues(fn)
        return make_private(rdd, self._budget_accountant, None)

    def flat_map(self, fn: Callable) -> 'PrivateRDD':
        """A Spark flatMap equivalent.

        Keeps track of privacy_id for each element.
        The output PrivateRDD has the same BudgetAccountant as self.
        """
        # Assumes that `self._rdd` consists of tuples `(privacy_id, element)`
        # and transforms each `element` according to the supplied function `fn`.
        rdd = self._rdd.flatMapValues(fn)
        return make_private(rdd, self._budget_accountant, None)

    def sum(self, sum_params: aggregate_params.SumParams) -> RDD:
        """Computes DP sum.

        Args:
            sum_params: parameters for calculation
        """

        ops = pipeline_dp.SparkRDDOperations()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, ops)

        params = pipeline_dp.AggregateParams(
            noise_kind=sum_params.noise_kind,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.
            max_contributions_per_partition,
            low=sum_params.low,
            high=sum_params.high,
            public_partitions=sum_params.public_partitions,
            budget_weight=sum_params.budget_weight)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: sum_params.partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: sum_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(self._rdd, params, data_extractors)

        return dp_result


def make_private(rdd: RDD,
                 budget_accountant: budget_accounting.BudgetAccountant,
                 privacy_id_extractor: Callable) -> PrivateRDD:
    """A factory method for PrivateRDD instance creation."""
    prdd = PrivateRDD(rdd, budget_accountant, privacy_id_extractor)
    return prdd
