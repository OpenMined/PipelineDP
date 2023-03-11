# Copyright 2022 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pyspark import RDD
from typing import Callable, Optional

import pipeline_dp
from pipeline_dp import aggregate_params, budget_accounting


class PrivateRDD:
    """A Spark RDD counterpart.

    PrivateRDD guarantees that only data that has been aggregated 
    in a DP manner, using no more than the specified privacy 
    budget, can be extracted from it through its API.

    PrivateRDD keeps a `privacy_id` for each element
    in order to guarantee correct DP computations.
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
        The output PrivateRDD has the same BudgetAccountant as this one.
        """
        # Assumes that `self._rdd` consists of tuples `(privacy_id, element)`
        # and transforms each `element` according to the supplied function `fn`.
        rdd = self._rdd.mapValues(fn)
        return make_private(rdd, self._budget_accountant, None)

    def flat_map(self, fn: Callable) -> 'PrivateRDD':
        """A Spark flatMap equivalent.

        Keeps track of privacy_id for each element.
        The output PrivateRDD has the same BudgetAccountant as this one.
        """
        # Assumes that `self._rdd` consists of tuples `(privacy_id, element)`
        # and transforms each `element` according to the supplied function `fn`.
        rdd = self._rdd.flatMapValues(fn)
        return make_private(rdd, self._budget_accountant, None)

    def variance(
        self,
        variance_params: aggregate_params.VarianceParams,
        public_partitions=None,
        out_explain_computaton_report: Optional[
            pipeline_dp.ExplainComputationReport] = None
    ) -> RDD:
        """Computes a DP variance.

        Args:
            variance_params: parameters for calculation
            public_partitions: A collection of partition keys that will be
              present in the result. Optional. If not provided, partitions will be selected in a DP
              manner.
            out_explain_computaton_report: an output argument, if specified,
              it will contain the Explain Computation report for this
              aggregation. For more details see the docstring to
              report_generator.py.
        """

        backend = pipeline_dp.SparkRDDBackend(self._rdd.context)
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=variance_params.noise_kind,
            metrics=[pipeline_dp.Metrics.VARIANCE],
            max_partitions_contributed=variance_params.
            max_partitions_contributed,
            max_contributions_per_partition=variance_params.
            max_contributions_per_partition,
            min_value=variance_params.min_value,
            max_value=variance_params.max_value,
            budget_weight=variance_params.budget_weight,
            contribution_bounds_already_enforced=variance_params.
            contribution_bounds_already_enforced)

        value_extractor_needed = not variance_params.contribution_bounds_already_enforced
        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: variance_params.partition_extractor(x[
                1]),
            privacy_id_extractor=self._get_privacy_id_extractor(
                params.contribution_bounds_already_enforced),
            value_extractor=lambda x: variance_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(
            self._rdd,
            params,
            data_extractors,
            public_partitions,
            out_explain_computaton_report=out_explain_computaton_report)
        # dp_result : (partition_key, (variance=dp_variance))

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - variance. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.variance,
                                       "Extract variance")
        # dp_result : (partition_key, dp_variance)

        return dp_result

    def mean(
        self,
        mean_params: aggregate_params.MeanParams,
        public_partitions=None,
        out_explain_computaton_report: Optional[
            pipeline_dp.ExplainComputationReport] = None
    ) -> RDD:
        """Computes a DP mean.

        Args:
            mean_params: parameters for calculation
            public_partitions: A collection of partition keys that will be
              present in the result. Optional. If not provided, partitions will
              be selected in a DP manner.
            out_explain_computaton_report: an output argument, if specified,
              it will contain the Explain Computation report for this
              aggregation. For more details see the docstring to
              report_generator.py.
        """
        backend = pipeline_dp.SparkRDDBackend(self._rdd.context)
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=mean_params.noise_kind,
            metrics=[pipeline_dp.Metrics.MEAN],
            max_partitions_contributed=mean_params.max_partitions_contributed,
            max_contributions_per_partition=mean_params.
            max_contributions_per_partition,
            min_value=mean_params.min_value,
            max_value=mean_params.max_value,
            budget_weight=mean_params.budget_weight,
            contribution_bounds_already_enforced=mean_params.
            contribution_bounds_already_enforced)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: mean_params.partition_extractor(x[1]),
            privacy_id_extractor=self._get_privacy_id_extractor(
                params.contribution_bounds_already_enforced),
            value_extractor=lambda x: mean_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(
            self._rdd,
            params,
            data_extractors,
            public_partitions,
            out_explain_computaton_report=out_explain_computaton_report)
        # dp_result : (partition_key, (mean=dp_mean))

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - mean. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.mean,
                                       "Extract mean")
        # dp_result : (partition_key, dp_mean)

        return dp_result

    def sum(
        self,
        sum_params: aggregate_params.SumParams,
        public_partitions=None,
        out_explain_computaton_report: Optional[
            pipeline_dp.ExplainComputationReport] = None
    ) -> RDD:
        """Computes a DP sum.

        Args:
            sum_params: parameters for calculation
            public_partitions: A collection of partition keys that will be
               present in the result. Optional. If not provided, partitions will
               be selected in a DP manner.
            out_explain_computaton_report: an output argument, if specified,
              it will contain the Explain Computation report for this
              aggregation. For more details see the docstring to
              report_generator.py.
        """
        backend = pipeline_dp.SparkRDDBackend(self._rdd.context)
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=sum_params.noise_kind,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=sum_params.max_partitions_contributed,
            max_contributions_per_partition=sum_params.
            max_contributions_per_partition,
            min_value=sum_params.min_value,
            max_value=sum_params.max_value,
            budget_weight=sum_params.budget_weight,
            contribution_bounds_already_enforced=sum_params.
            contribution_bounds_already_enforced)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: sum_params.partition_extractor(x[1]),
            privacy_id_extractor=self._get_privacy_id_extractor(
                params.contribution_bounds_already_enforced),
            value_extractor=lambda x: sum_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(
            self._rdd,
            params,
            data_extractors,
            public_partitions,
            out_explain_computaton_report=out_explain_computaton_report)
        # dp_result : (partition_key, (sum=dp_sum))

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - sum. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.sum,
                                       "Extract sum")
        # dp_result : (partition_key, dp_sum)

        return dp_result

    def count(
        self,
        count_params: aggregate_params.CountParams,
        public_partitions=None,
        out_explain_computaton_report: Optional[
            pipeline_dp.ExplainComputationReport] = None
    ) -> RDD:
        """Computes a DP count.

        Args:
            count_params: parameters for calculation
            public_partitions: A collection of partition keys that will be
              present in the result. Optional. If not provided, partitions will
              be selected in a DP manner.
            out_explain_computaton_report: an output argument, if specified,
              it will contain the Explain Computation report for this
              aggregation. For more details see the docstring to
              report_generator.py.
        """
        backend = pipeline_dp.SparkRDDBackend(self._rdd.context)
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=count_params.noise_kind,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=count_params.max_partitions_contributed,
            max_contributions_per_partition=count_params.
            max_contributions_per_partition,
            budget_weight=count_params.budget_weight,
            contribution_bounds_already_enforced=count_params.
            contribution_bounds_already_enforced)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: count_params.partition_extractor(x[1]
                                                                          ),
            privacy_id_extractor=self._get_privacy_id_extractor(
                params.contribution_bounds_already_enforced),
            value_extractor=lambda x: None)

        dp_result = dp_engine.aggregate(
            self._rdd,
            params,
            data_extractors,
            public_partitions,
            out_explain_computaton_report=out_explain_computaton_report)
        # dp_result : (partition_key, (count=dp_count))

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - count. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.count,
                                       "Extract count")
        # dp_result : (partition_key, dp_count)

        return dp_result

    def privacy_id_count(
        self,
        privacy_id_count_params: aggregate_params.PrivacyIdCountParams,
        public_partitions=None,
        out_explain_computaton_report: Optional[
            pipeline_dp.ExplainComputationReport] = None
    ) -> RDD:
        """Computes a DP Privacy ID count.

        Args:
            privacy_id_count_params: parameters for calculation
            public_partitions: A collection of partition keys that will be
              present in the result. Optional. If not provided, partitions will
              be selected in a DP manner.
            out_explain_computaton_report: an output argument, if specified,
              it will contain the Explain Computation report for this
              aggregation. For more details see the docstring to
              report_generator.py.
        """
        backend = pipeline_dp.SparkRDDBackend(self._rdd.context)
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=privacy_id_count_params.noise_kind,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            max_partitions_contributed=privacy_id_count_params.
            max_partitions_contributed,
            max_contributions_per_partition=1,
            contribution_bounds_already_enforced=privacy_id_count_params.
            contribution_bounds_already_enforced)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: privacy_id_count_params.
            partition_extractor(x[1]),
            privacy_id_extractor=self._get_privacy_id_extractor(
                params.contribution_bounds_already_enforced),
            # PrivacyIdCount ignores values.
            value_extractor=lambda x: None)

        dp_result = dp_engine.aggregate(
            self._rdd,
            params,
            data_extractors,
            public_partitions,
            out_explain_computaton_report=out_explain_computaton_report)
        # dp_result : (partition_key, (privacy_id_count=dp_privacy_id_count))

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - privacy id count. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.privacy_id_count,
                                       "Extract privacy id count")
        # dp_result : (partition_key, dp_privacy_id_count)

        return dp_result

    def select_partitions(
            self,
            select_partitions_params: aggregate_params.SelectPartitionsParams,
            partition_extractor: Callable) -> RDD:
        """Computes a collection of partition keys in a DP manner.

        Args:
            select_partitions_params: parameters for calculation
            partition_extractor: function for extracting partition key from each input element
        """

        backend = pipeline_dp.SparkRDDBackend(self._rdd.context)
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.SelectPartitionsParams(
            max_partitions_contributed=select_partitions_params.
            max_partitions_contributed)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0])

        return dp_engine.select_partitions(self._rdd, params, data_extractors)

    def _get_privacy_id_extractor(self,
                                  contribution_bounds_already_enforced: bool):
        if contribution_bounds_already_enforced:
            # Privacy ids are not needed when contribution bounding already
            # enforced.
            return None
        return lambda x: x[0]


def make_private(rdd: RDD,
                 budget_accountant: budget_accounting.BudgetAccountant,
                 privacy_id_extractor: Callable) -> PrivateRDD:
    """A factory method for creating PrivateRDDs."""
    prdd = PrivateRDD(rdd, budget_accountant, privacy_id_extractor)
    return prdd
