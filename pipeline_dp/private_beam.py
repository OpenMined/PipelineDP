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


class Mean(PrivatePTransform):
    """Transform class for performing DP Mean on PrivatePCollection."""

    def __init__(self,
                 mean_params: aggregate_params.MeanParams,
                 label: Optional[str] = None):
        super().__init__(return_anonymized=True, label=label)
        self._mean_params = mean_params

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
            max_value=self._mean_params.max_value,
            public_partitions=self._mean_params.public_partitions)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._mean_params.partition_extractor(
                x[1]),
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: self._mean_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(pcol, params, data_extractors)
        # dp_result : (partition_key, [dp_sum])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - mean. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.mean,
                                       "Extract mean")
        # dp_result : (partition_key, dp_sum)

        return dp_result


class Sum(PrivatePTransform):
    """Transform class for performing DP Sum on a PrivatePCollection."""

    def __init__(self,
                 sum_params: aggregate_params.SumParams,
                 label: Optional[str] = None):
        super().__init__(return_anonymized=True, label=label)
        self._sum_params = sum_params

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
            max_value=self._sum_params.max_value,
            public_partitions=self._sum_params.public_partitions)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._sum_params.partition_extractor(
                x[1]),
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: self._sum_params.value_extractor(x[1]))

        dp_result = dp_engine.aggregate(pcol, params, data_extractors)
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
                 label: Optional[str] = None):
        super().__init__(return_anonymized=True, label=label)
        self._count_params = count_params

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._count_params.noise_kind,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=self._count_params.
            max_partitions_contributed,
            max_contributions_per_partition=self._count_params.
            max_contributions_per_partition,
            public_partitions=self._count_params.public_partitions)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._count_params.
            partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0],
            # Count calculates the number of elements per partition key and
            # doesn't use value extractor.
            value_extractor=lambda x: None)

        dp_result = dp_engine.aggregate(pcol, params, data_extractors)
        # dp_result : (partition_key, [dp_count])

        # aggregate() returns a namedtuple of metrics for each partition key.
        # Here is only one metric - count. Extract it from the list.
        dp_result = backend.map_values(dp_result, lambda v: v.count,
                                       "Extract sum")
        # dp_result : (partition_key, dp_count)

        return dp_result


class PrivacyIdCount(PrivatePTransform):
    """Transform class for performing a DP Privacy ID Count on a PrivatePCollection."""

    def __init__(self,
                 privacy_id_count_params: aggregate_params.PrivacyIdCountParams,
                 label: Optional[str] = None):
        super().__init__(return_anonymized=True, label=label)
        self._privacy_id_count_params = privacy_id_count_params

    def expand(self, pcol: pvalue.PCollection) -> pvalue.PCollection:
        backend = pipeline_dp.BeamBackend()
        dp_engine = pipeline_dp.DPEngine(self._budget_accountant, backend)

        params = pipeline_dp.AggregateParams(
            noise_kind=self._privacy_id_count_params.noise_kind,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            max_partitions_contributed=self._privacy_id_count_params.
            max_partitions_contributed,
            max_contributions_per_partition=1,
            public_partitions=self._privacy_id_count_params.public_partitions)

        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: self._privacy_id_count_params.
            partition_extractor(x[1]),
            privacy_id_extractor=lambda x: x[0],
            # PrivacyIdCount ignores values.
            value_extractor=lambda x: None)

        dp_result = dp_engine.aggregate(pcol, params, data_extractors)
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
    @abc.abstractmethod
    def request_budget(self,
        budget_accountant: budget_accounting.BudgetAccountant):

    @abc.abstractmethod
    def add_private_input(self, accumulator, input):
        pass

    @abc.abstractmethod
    def extract_private_output(self, accumulator):
        pass

    @abc.abstractmethod
    def metric_names(self):
        pass


class CombineFnCombiner(pipeline_dp.CustomCombiner):
    def __init__(self, private_combine_fn:PrivateCombineFn):
        self._private_combine_fn = private_combine_fn

    def create_accumulator(self, values):
        """Creates accumulator from 'values'."""
        accumulator = self._private_combine_fn.create_accumulator()
        for v in values:
            self._private_combine_fn.add_private_input(accumulator, input)

    def merge_accumulators(self, accumulator1, accumulator2):
        """Merges the accumulators and returns accumulator."""
        return self._private_combine_fn.merge_accumulators(accumulator1, accumulator2)

    def compute_metrics(self, accumulator):
        """Computes and returns the result of aggregation."""
        return self._private_combine_fn.extract_private_output(accumulator)

    def metrics_names(self) -> typing.List[str]:
        """Return the list of names of the metrics this combiner computes"""
        return self._private_combine_fn.metric_names()

    def request_budget(self,
                       budget_accountant: budget_accounting.BudgetAccountant):
        return self._private_combine_fn.request_budget(budget_accountant)


@dataclasses.dataclass
class CombinePerKeyParams:
    max_partitions_contributed: int
    max_contributions_per_partition: int
    budget_weight: float = 1
    min_value: float = None  # ?
    max_value: float = None  # ?
    public_partitions: typing.Any = None


class CombinePerKey(PrivatePTransform):
    """Transform class for performing a FlatMap on a PrivatePCollection."""

    def __init__(self,
                 combine_fn: PrivateCombineFn,
                 params:CombinePerKeyParams,
                 label: Optional[str] = None):
        super().__init__(return_anonymized=True, label=label)
        self._combine_fn = combine_fn
        self._params = params

    def expand(self, pcol: pvalue.PCollection):
        combiner = CombineFnCombiner(self._combine_fn)
        aggregate_params = None # todo
        dp_engine = None # todo
        data_extractors = None # todo

        return dp_engine.aggregate(pcol, aggregate_params, data_extractors)

