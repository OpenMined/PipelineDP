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
"""Combiners for DP utility analysis.

These combiners are in the same format of pipeline.combiners but without DP
noises and computation.
"""
import abc
import copy
from typing import Iterable, Sized, Tuple, List

import pipeline_dp
import numpy as np
from collections import namedtuple


class RawCountCombiner(pipeline_dp.combiners.Combiner):
    """Combiner for computing DP Count.

    The type of the accumulator is int, which represents count of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = int

    def create_accumulator(self, values: Sized) -> AccumulatorType:
        return len(values)

    def merge_accumulators(self, count1: AccumulatorType,
                           count2: AccumulatorType):
        return count1 + count2

    def compute_metrics(self, count: AccumulatorType) -> float:
        return count

    def metrics_names(self) -> List[str]:
        return ['non_private_count']


class RawPrivacyIdCountCombiner(pipeline_dp.combiners.Combiner):
    """Combiner for computing DP privacy id count.
    The type of the accumulator is int, which represents count of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = int

    def create_accumulator(self, values: Sized) -> AccumulatorType:
        return 1 if values else 0

    def merge_accumulators(self, accumulator1: AccumulatorType,
                           accumulator2: AccumulatorType):
        return accumulator1 + accumulator2

    def compute_metrics(self, accumulator: AccumulatorType) -> float:
        return accumulator

    def metrics_names(self) -> List[str]:
        return ['non_private_privacy_id_count']


class RawSumCombiner(pipeline_dp.combiners.Combiner):
    """Combiner for computing dp sum.

    the type of the accumulator is int, which represents sum of the elements
    in the dataset for which this accumulator is computed.
    """
    AccumulatorType = float

    def create_accumulator(self, values: Iterable[float]) -> 'AccumulatorType':
        return sum(values)

    def merge_accumulators(self, sum1: AccumulatorType, sum2: AccumulatorType):
        return sum1 + sum2

    def compute_metrics(self, accumulator: AccumulatorType) -> float:
        return accumulator

    def metrics_names(self) -> List[str]:
        return ['non_private_sum']


MeanTuple = namedtuple('MeanTuple', ['count', 'sum', 'mean'])


class RawMeanCombiner(pipeline_dp.combiners.Combiner):
    """Combiner for computing DP Mean. Also returns sum and count in addition to
    the mean.
    The type of the accumulator is a tuple(count: int, sum: float) that holds
    the count and sum of elements in the dataset for which this accumulator is
    computed.
    """
    AccumulatorType = Tuple[int, float]

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        return len(values), sum(values)

    def merge_accumulators(self, accum1: AccumulatorType,
                           accum2: AccumulatorType):
        count1, sum1 = accum1
        count2, sum2 = accum2
        return count1 + count2, sum1 + sum2

    def compute_metrics(self, accum: AccumulatorType) -> namedtuple:
        count, summary = accum
        return MeanTuple(count=count,
                         sum=summary,
                         mean=summary / count if count else None)

    def metrics_names(self) -> List[str]:
        return ['non_private_mean']


VarianceTuple = namedtuple('VarianceTuple', ['count', 'sum', 'mean', 'variance'])


class RawVarianceCombiner(pipeline_dp.combiners.Combiner):
    """Combiner for computing DP Variance. Also returns sum, count and mean in addition to
    the variance.
    The type of the accumulator is a tuple(count: int, sum: float) that holds
    the count and sum of elements in the dataset for which this accumulator is
    computed.
    """
    AccumulatorType = Tuple[int, float, float]

    def create_accumulator(self, values: Iterable[float]) -> AccumulatorType:
        return len(values), sum(values), sum(value ** 2 for value in values)

    def merge_accumulators(self, accum1: AccumulatorType,
                           accum2: AccumulatorType):
        count1, sum1, sum_of_squares1 = accum1
        count2, sum2, sum_of_squares2 = accum2
        return count1 + count2, sum1 + sum2, sum_of_squares1 + sum_of_squares2

    def compute_metrics(self, accum: AccumulatorType) -> namedtuple:
        count, sum, sum_of_squares = accum
        return VarianceTuple(count=count,
                             sum=sum,
                             mean=sum / count if count else None,
                             variance=sum_of_squares / count - sum / count if count else None)

    def metrics_names(self) -> List[str]:
        return ['non_private_mean']


class CompoundCombiner(pipeline_dp.combiners.Combiner):
    """Combiner for computing a set of dp aggregations.

    CompoundCombiner contains one or more combiners of other types for computing multiple metrics.
    For example it can contain [CountCombiner, SumCombiner].
    CompoundCombiner delegates all operations to the internal combiners.

    The type of the accumulator is a tuple that contains accumulators from internal combiners.
    """

    AccumulatorType = Tuple

    def __init__(self, combiners: Iterable['Combiner']):
        self._combiners = combiners
        self._metrics_to_compute = []
        for combiner in self._combiners:
            self._metrics_to_compute.extend(combiner.metrics_names())
        if len(self._metrics_to_compute) != len(set(self._metrics_to_compute)):
            raise ValueError(
                f"two combiners in {combiners} cannot compute the same metrics")
        self._MetricsTuple = namedtuple('MetricsTuple',
                                        self._metrics_to_compute)

    def create_accumulator(self, values) -> AccumulatorType:
        return tuple(
            combiner.create_accumulator(values) for combiner in self._combiners)

    def merge_accumulators(self, accumulator1: AccumulatorType,
                           accumulator2: AccumulatorType) -> AccumulatorType:
        merged_accumulators = []
        for combiner, acc1, acc2 in zip(self._combiners, accumulator1,
                                        accumulator2):
            merged_accumulators.append(combiner.merge_accumulators(acc1, acc2))
        return tuple(merged_accumulators)

    def compute_metrics(self, accumulator: AccumulatorType) -> list:
        metrics = []
        for combiner, acc in zip(self._combiners, accumulator):
            metrics.append(combiner.compute_metrics(acc))
        return metrics

    def metrics_names(self) -> List[str]:
        return self._metrics_to_compute


def create_compound_combiner(
        metrics: pipeline_dp.aggregate_params.Metrics) -> CompoundCombiner:
    combiners = []
    if pipeline_dp.Metrics.COUNT in metrics:
        combiners.append(RawCountCombiner())
    if pipeline_dp.Metrics.SUM in metrics:
        combiners.append(RawSumCombiner())
    if pipeline_dp.Metrics.PRIVACY_ID_COUNT in metrics:
        combiners.append(RawPrivacyIdCountCombiner())
    if pipeline_dp.Metrics.MEAN in metrics:
        combiners.append(RawMeanCombiner())
    if pipeline_dp.Metrics.VARIANCE in metrics:
        combiners.append(RawVarianceCombiner())
    return CompoundCombiner(combiners)
