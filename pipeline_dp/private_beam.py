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
import abc
import dataclasses
import typing
from apache_beam.transforms import ptransform
from abc import abstractmethod
from typing import Callable, Optional
from apache_beam import pvalue
import apache_beam as beam

import pipeline_dp
from pipeline_dp import aggregate_params, budget_accounting


class PrivatePTransform(ptransform.PTransform):
    """Abstract class for PrivatePTransforms."""

    def __init__(self, return_anonymized: bool, label: Optional[str] = None):
        super().__init__(label)
        self._return_anonymized = return_anonymized
        self._budget_accountant = None

    def set_additional_parameters(
            self, budget_accountant: budget_accounting.BudgetAccountant):
        """Sets the additional parameters needed for the private transform."""
        self._budget_accountant = budget_accountant

    def _create_dp_engine(self):
        backend = pipeline_dp.BeamBackend()
        return backend, pipeline_dp.DPEngine(self._budget_accountant, backend)

    def __rrshift__(self, label):
        self.label = label
        return self

    @abstractmethod
    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        pass


class PrivatePCollection:
    """Private counterpart for PCollection.

    PrivatePCollection guarantees that only data that has been aggregated 
    in a DP manner, using no more than the specified
    privacy budget, can be extracted from it using PrivatePTransforms."""

    def __init__(self, pcol: pvalue.PCollection,
                 budget_accountant: budget_accounting.BudgetAccountant):
        self._pcol = pcol
        self._budget_accountant = budget_accountant

    def __or__(self, private_transform: PrivatePTransform):
        if not isinstance(private_transform, PrivatePTransform):
            raise TypeError(
                "private_transform should be of type PrivatePTransform but is "
                + "%s", private_transform)

        private_transform.set_additional_parameters(
            budget_accountant=self._budget_accountant)
        transformed = self._pcol.pipeline.apply(private_transform, self._pcol)

        return (transformed if private_transform._return_anonymized else
                (PrivatePCollection(transformed, self._budget_accountant)))


class MakePrivate(PrivatePTransform):
    """Transform class for creating a PrivatePCollection."""

    def __init__(self,
                 budget_accountant: budget_accounting.BudgetAccountant,
                 privacy_id_extractor: Callable,
                 label: Optional[str] = None):
        super().__init__(return_anonymized=False, label=label)
        self._budget_accountant = budget_accountant
        self._privacy_id_extractor = privacy_id_extractor

    def expand(self, pcol: pvalue.PCollection):
        pcol = pcol | "Extract privacy id" >> beam.Map(
            lambda x: (self._privacy_id_extractor(x), x))
        return PrivatePCollection(pcol, self._budget_accountant)


class Variance(PrivatePTransform):
    """Transform class for performing DP Variance on PrivatePCollection."""

    def __init__(self,
                 variance_params: aggregate_params.VarianceParams,
                 label: Optional[str] = None,
                 public_partitions=None,
                 out_explain_computaton_report: Optional[
                     pipeline_dp.ExplainComputationReport] = None):
        """Initialize.

         Args:
             variance_params: parameters for calculation
             public_partitions: A collection of partition keys that will be
               present in the result. Optional. If not provided, partitions will
               be selected in a DP manner.
             out_explain_computaton_report: an output argument, if specified,
                it will contain the Explain Computation report for this
                aggregation. For more details see the docstring to
                report_generator.py.
         """
        super().__init__(return_anonymized=True, label=label)
        self._variance_params = variance_params
        self._public_partitions = public_partitions
        self._explain_computaton_report = out_explain_computaton_report

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._variance_params.noise_kind,
            metrics=[pipeline_dp.Metrics.VARIANCE],
            max_partitions_contributed=self._variance_params.
            max_partitions_contributed,
            max_contributions_per_partition=self._variance_params.
            max_contributions_per_partition,
            min_value=self._variance_params.min_value,
            max_value=self._variance_params.max_value)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._variance_params.
            partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: self._variance_params.value_extractor(x[1]
                                                                           ))

        dp_result = dp_engine.aggregate(
            pcol,
            params,
            data_extractors,
            self._public_partitions,
            out_explain_computaton_report=self._explain_computaton_report)
        # dp_result : (partition_key, [dp_variance])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - variance. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.variance,
                                       "Extract variance")
        # dp_result : (partition_key, dp_variance)

        return dp_result


class Mean(PrivatePTransform):
    """Transform class for performing DP Mean on PrivatePCollection."""

    def __init__(self,
                 mean_params: aggregate_params.MeanParams,
                 label: Optional[str] = None,
                 public_partitions=None,
                 out_explain_computaton_report: Optional[
                     pipeline_dp.ExplainComputationReport] = None):
        """Initialize

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
        super().__init__(return_anonymized=True, label=label)
        self._mean_params = mean_params
        self._public_partitions = public_partitions
        self._explain_computaton_report = out_explain_computaton_report

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._mean_params.noise_kind,
            metrics=[pipeline_dp.Metrics.MEAN],
            max_partitions_contributed=self._mean_params.
            max_partitions_contributed,
            max_contributions_per_partition=self._mean_params.
            max_contributions_per_partition,
            min_value=self._mean_params.min_value,
            max_value=self._mean_params.max_value)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._mean_params.partition_extractor(
                x[1]),
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: self._mean_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(
            pcol,
            params,
            data_extractors,
            self._public_partitions,
            out_explain_computaton_report=self._explain_computaton_report)
        # dp_result : (partition_key, [dp_mean])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - mean. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.mean,
                                       "Extract mean")
        # dp_result : (partition_key, dp_mean)

        return dp_result


class Sum(PrivatePTransform):
    """Transform class for performing DP Sum on a PrivatePCollection."""

    def __init__(self,
                 sum_params: aggregate_params.SumParams,
                 label: Optional[str] = None,
                 public_partitions=None,
                 out_explain_computaton_report: Optional[
                     pipeline_dp.ExplainComputationReport] = None):
        """Initialize.

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
        super().__init__(return_anonymized=True, label=label)
        self._sum_params = sum_params
        self._public_partitions = public_partitions
        self._explain_computaton_report = out_explain_computaton_report

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._sum_params.noise_kind,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=self._sum_params.
            max_partitions_contributed,
            max_contributions_per_partition=self._sum_params.
            max_contributions_per_partition,
            min_value=self._sum_params.min_value,
            max_value=self._sum_params.max_value)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._sum_params.partition_extractor(
                x[1]),
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: self._sum_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(
            pcol,
            params,
            data_extractors,
            self._public_partitions,
            out_explain_computaton_report=self._explain_computaton_report)
        # dp_result : (partition_key, [dp_sum])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - sum. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.sum,
                                       "Extract sum")
        # dp_result : (partition_key, dp_sum)

        return dp_result


class Count(PrivatePTransform):
    """Transform class for performing DP Count on a PrivatePCollection."""

    def __init__(self,
                 count_params: aggregate_params.CountParams,
                 label: Optional[str] = None,
                 public_partitions=None,
                 out_explain_computaton_report: Optional[
                     pipeline_dp.ExplainComputationReport] = None):
        """Initialize.

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
        super().__init__(return_anonymized=True, label=label)
        self._count_params = count_params
        self._public_partitions = public_partitions
        self._explain_computaton_report = out_explain_computaton_report

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._count_params.noise_kind,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=self._count_params.
            max_partitions_contributed,
            max_contributions_per_partition=self._count_params.
            max_contributions_per_partition)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._count_params.
            partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0],
            # Count calculates the number of elements per partition key and
            # doesn't use value extractor.
            value_extractor=lambda x: None)

        dp_result = dp_engine.aggregate(
            pcol,
            params,
            data_extractors,
            self._public_partitions,
            out_explain_computaton_report=self._explain_computaton_report)
        # dp_result : (partition_key, [dp_count])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - count. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.count,
                                       "Extract count")
        # dp_result : (partition_key, dp_count)

        return dp_result


class PrivacyIdCount(PrivatePTransform):
    """Transform class for performing a DP Privacy ID Count on a PrivatePCollection."""

    def __init__(self,
                 privacy_id_count_params: aggregate_params.PrivacyIdCountParams,
                 label: Optional[str] = None,
                 public_partitions=None,
                 out_explain_computaton_report: Optional[
                     pipeline_dp.ExplainComputationReport] = None):
        """Initialize.

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
        super().__init__(return_anonymized=True, label=label)
        self._privacy_id_count_params = privacy_id_count_params
        self._public_partitions = public_partitions
        self._explain_computaton_report = out_explain_computaton_report

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._privacy_id_count_params.noise_kind,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            max_partitions_contributed=self._privacy_id_count_params.
            max_partitions_contributed,
            max_contributions_per_partition=1)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._privacy_id_count_params.
            partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0],
            # PrivacyIdCount ignores values.
            value_extractor=lambda x: None)

        dp_result = dp_engine.aggregate(
            pcol,
            params,
            data_extractors,
            self._public_partitions,
            out_explain_computaton_report=self._explain_computaton_report)
        # dp_result : (partition_key, [dp_privacy_id_count])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - privacy_id_count. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.privacy_id_count,
                                       "Extract privacy_id_count")
        # dp_result : (partition_key, dp_privacy_id_count)

        return dp_result


class SelectPartitions(PrivatePTransform):
    """Transform class for computing a collection of partition keys using DP."""

    def __init__(
            self,
            select_partitions_params: aggregate_params.SelectPartitionsParams,
            partition_extractor: Callable, label: Optional[str]):
        super().__init__(return_anonymized=True, label=label)
        self._select_partitions_params = select_partitions_params
        self._partition_extractor = partition_extractor

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0])

        dp_result = dp_engine.select_partitions(pcol,
                                                self._select_partitions_params,
                                                data_extractors)

        return dp_result


class Map(PrivatePTransform):
    """Transform class for performing a Map on a PrivatePCollection."""

    def __init__(self, fn: Callable, label: Optional[str] = None):
        super().__init__(return_anonymized=False, label=label)
        self._fn = fn

    def expand(self, pcol: pvalue.PCollection):
        return pcol | "map values" >> beam.Map(lambda x: (x[0], self._fn(x[1])))


class FlatMap(PrivatePTransform):
    """Transform class for performing a FlatMap on a PrivatePCollection."""

    class _FlattenValues(beam.DoFn):
        """Inner class for flattening values of key value pair.
        Flattens (1, (2,3,4)) into ((1,2), (1,3), (1,4))"""

        def __init__(self, map_fn: Callable):
            self._map_fn = map_fn

        def process(self, row):
            key = row[0]
            values = self._map_fn(row[1])
            for value in values:
                yield key, value

    def __init__(self, fn: Callable, label: Optional[str] = None):
        super().__init__(return_anonymized=False, label=label)
        self._fn = fn

    def expand(self, pcol: pvalue.PCollection):
        return pcol | "flatten values" >> beam.ParDo(
            FlatMap._FlattenValues(map_fn=self._fn))


class PrivateCombineFn(beam.CombineFn):
    """Base class for custom private CombinerFns.

    Warning: this is an experimental API. It might not work properly and it
    might be changed/removed without any notifications.

    Custom private CombinerFns are implemented by PipelineDP users and they
    allow to add custom DP aggregations for extending the PipelineDP
    functionality.

    The responsibility of PrivateCombineFn:
      1.Implement DP mechanism in `extract_private_output()`.
      2.If needed implement contribution bounding in
    `add_input_for_private_output()`.

    Warning: this is an advanced feature that can break differential privacy
    guarantees if not implemented correctly.
    """

    @abc.abstractmethod
    def add_input_for_private_output(self, accumulator, input):
        """Add input, which contributes to private output.

         This is a DP counterpart of `add_input()`. The same CombinerFn can
         have both in order to be able to compute both DP and not-DP
         aggregations.

         Typically, this function should perform input clipping to ensure
         differential privacy.
         """

    @abc.abstractmethod
    def extract_private_output(self, accumulator,
                               budget: budget_accounting.MechanismSpec):
        """Computes private output.

        'budget' is the object which returned from 'request_budget()'.
        """

    @abc.abstractmethod
    def request_budget(
        self, budget_accountant: budget_accounting.BudgetAccountant
    ) -> budget_accounting.MechanismSpec:
        """Requests the budget.

        It is called by PipelineDP during the construction of the computations.
        The custom combiner can request a DP budget by calling
        'budget_accountant.request_budget()'. The budget object needs to be
        returned. It will be serialized and distributed to the workers together
        with 'self'.

        Warning: do not store 'budget_accountant' in 'self'. It is assumed to
        live in the driver process.
        """

    def set_aggregate_params(self,
                             aggregate_params: pipeline_dp.AggregateParams):
        """Sets aggregate parameters

        The custom combiner can optionally use it for own DP parameter
        computations.
        """
        self._aggregate_params = aggregate_params


class _CombineFnCombiner(pipeline_dp.CustomCombiner):

    def __init__(self, private_combine_fn: PrivateCombineFn):
        self._private_combine_fn = private_combine_fn

    def create_accumulator(self, values):
        """Creates accumulator from 'values'."""
        accumulator = self._private_combine_fn.create_accumulator()
        for v in values:
            accumulator = self._private_combine_fn.add_input_for_private_output(
                accumulator, v)
        return accumulator

    def merge_accumulators(self, accumulator1, accumulator2):
        """Merges the accumulators and returns accumulator."""
        return self._private_combine_fn.merge_accumulators(
            [accumulator1, accumulator2])

    def compute_metrics(self, accumulator):
        """Computes and returns the result of aggregation."""
        return self._private_combine_fn.extract_private_output(
            accumulator, self._budget)

    def explain_computation(self) -> str:
        # TODO: implement
        return "Explain computations for PrivateCombineFn not implemented."

    def request_budget(self,
                       budget_accountant: budget_accounting.BudgetAccountant):
        self._budget = self._private_combine_fn.request_budget(
            budget_accountant)

    def set_aggregate_params(self, aggregate_params):
        self._private_combine_fn.set_aggregate_params(aggregate_params)


@dataclasses.dataclass
class CombinePerKeyParams:
    """Specifies parameters for private PTransform CombinePerKey.

     Args:
       max_partitions_contributed: A bound on the number of partitions to which one
         unit of privacy (e.g., a user) can contribute.
       max_contributions_per_partition: A bound on the number of times one unit of
         privacy (e.g. a user) can contribute to a partition.
       budget_weight: Relative weight of the privacy budget allocated to this
         aggregation.
       public_partitions: A collection of partition keys that will be present in
         the result. Optional. If not provided, partitions will be selected in a DP
         manner.
 """
    max_partitions_contributed: int
    max_contributions_per_partition: int
    budget_weight: float = 1
    public_partitions: typing.Any = None


class CombinePerKey(PrivatePTransform):
    """Transform class for performing a CombinePerKey on a PrivatePCollection.

    The assumption is that an input PrivatePCollection has elements of form
    (key, value). The elements of PrivatePCollection can be transformed with
    Map private transform.
    """

    def __init__(self,
                 combine_fn: PrivateCombineFn,
                 params: CombinePerKeyParams,
                 label: Optional[str] = None):
        super().__init__(return_anonymized=True, label=label)
        self._combine_fn = combine_fn
        self._params = params

    def expand(self, pcol: pvalue.PCollection):
        combiner = _CombineFnCombiner(self._combine_fn)
        aggregate_params = pipeline_dp.AggregateParams(
            metrics=None,
            max_partitions_contributed=self._params.max_partitions_contributed,
            max_contributions_per_partition=self._params.
            max_contributions_per_partition,
            custom_combiners=[combiner])

        backend, dp_engine = self._create_dp_engine()
        # Assumed elements format: (privacy_id, (partition_key, value))
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1][0],
            value_extractor=lambda x: x[1][1])

        dp_result = dp_engine.aggregate(pcol, aggregate_params, data_extractors)
        # dp_result : (partition_key, [combiner_result])

        # aggregate() returns a tuple with on 1 element per combiner.
        # Here is only one combiner. Extract it from the tuple.
        dp_result = backend.map_values(dp_result, lambda v: v[0],
                                       "Unnest tuple")
        # dp_result : (partition_key, result)

        return dp_result
