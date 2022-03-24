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
"""Accumulators for DP utility analysis.

These accumulators are in the same format of pipeline.accumulator but without DP
noises.
"""
import abc

from typing import Any, List, Sequence

import pipeline_dp


class CountAccumulator(pipeline_dp.accumulator.Accumulator):
    """Accumulator for count aggregation.

  Attributes:
    count: The count of input values.
  """

    def __init__(self, values: Sequence[Any]):
        self.count = len(values)

    def add_value(self, value: Any):
        """Adds 'value' to accumulate.

    Args:
      value: value to be added.
    """
        self.count += 1

    def add_accumulator(self,
                        accumulator: 'CountAccumulator') -> 'CountAccumulator':
        """Merges the accumulator to self and returns self.

    Args:
      accumulator: Another accumulator of the same type to merge.

    Returns:
      self
    """
        self._check_mergeable(accumulator)
        self.count += accumulator.count
        return self

    def compute_metrics(self) -> float:
        """Computes and returns the result of aggregation.

    Returns:
      Count of the values.
    """
        return self.count


class SumAccumulator(pipeline_dp.accumulator.Accumulator):
    """Accumulator for sum aggregation.

  Attributes:
    sum: The sum of the input values.
  """

    def __init__(self, values: Sequence[float]):
        self.sum = sum(values)

    def add_value(self, value: float):
        """Adds 'value' to accumulate.

    Args:
      value: value to be added.
    """
        self.sum += value

    def add_accumulator(self,
                        accumulator: 'SumAccumulator') -> 'SumAccumulator':
        """Merges the accumulator to self and returns self.

    Args:
      accumulator: Another accumulator of the same type to merge.

    Returns:
      self
    """
        self._check_mergeable(accumulator)
        self.sum += accumulator.sum
        return self

    def compute_metrics(self) -> float:
        """Computes and returns the result of aggregation.

    Returns:
      Sum of the values.
    """
        return self.sum


class PrivacyIdCountAccumulator(pipeline_dp.accumulator.Accumulator):
    """Accumulator for privacy id count aggregation.

  Should be constructed per (privacy_key, privacy_id)

  Attributes:
    count: The count of privacy ids in input values.
  """

    def __init__(self, values: Sequence[Any]):
        self.count = 1

    def add_value(self, value: Any) -> None:
        """Adds 'value' to accumulate.

    Args:
      value: value to be added.
    """

    def add_accumulator(
        self, accumulator: 'PrivacyIdCountAccumulator'
    ) -> 'PrivacyIdCountAccumulator':
        """Merges the accumulator to self and returns self.

    Args:
      accumulator: Another accumulator of the same type to merge.

    Returns:
      self
    """
        self._check_mergeable(accumulator)
        self.count += accumulator.count
        return self

    def compute_metrics(self) -> float:
        """Computes and returns the result of aggregation.

    Returns:
      Privacy id count of the values.
    """
        return self.count


class AccumulatorFactory(abc.ABC):
    """Abstract base class for all accumulator factories.

  Each concrete implementation of AccumulatorFactory creates Accumulator of
  the specific type.
  """

    @abc.abstractmethod
    def create(self,
               values: Sequence[Any]) -> pipeline_dp.accumulator.Accumulator:
        """Creates an accumulator for the corresponding input values.

    Args:
      values: The input values to compute accumulation.

    Returns:
      A new accumulator with values.
    """


def _create_accumulator_factories(
        metrics: Sequence[pipeline_dp.Metrics]) -> List[AccumulatorFactory]:
    """Creates accumulator factories for the given metrics.

  Args:
    metrics: All metrics to compute. Dupliate metrics will be seen as one.

  Returns:
    A list of accumulator factories corrsponding to the input metrics.
  """
    factories = []
    if pipeline_dp.Metrics.COUNT in metrics:
        factories.append(CountAccumulatorFactory())
    if pipeline_dp.Metrics.SUM in metrics:
        factories.append(SumAccumulatorFactory())
    if pipeline_dp.Metrics.PRIVACY_ID_COUNT in metrics:
        factories.append(PrivacyIdCountAccumulatorFactory())
    return factories


class CountAccumulatorFactory(AccumulatorFactory):

    def create(self, values: Sequence[Any]) -> CountAccumulator:
        del self  # unused, for implementing abstract class
        return CountAccumulator(values)


class SumAccumulatorFactory(AccumulatorFactory):

    def create(self, values: Sequence[Any]) -> SumAccumulator:
        del self  # unused, for implementing abstract class
        return SumAccumulator(values)


class PrivacyIdCountAccumulatorFactory(AccumulatorFactory):

    def create(self, values: Sequence[Any]) -> PrivacyIdCountAccumulator:
        del self  # unused, for implementing abstract class
        return PrivacyIdCountAccumulator(values)


class CompoundAccumulatorFactory(AccumulatorFactory):
    """Factory for creating CompoundAccumulator.

  CompoundAccumulatorFactory contains one or more AccumulatorFactories which
  create accumulators for specific metrics. These AccumulatorFactories are
  created based on pipeline_dp.AggregateParams.
  """

    def __init__(self, metrics: Sequence[pipeline_dp.Metrics]):
        self._accumulator_factories = _create_accumulator_factories(metrics)

    def create(self,
               values: Sequence[Any]) -> pipeline_dp.accumulator.Accumulator:
        accumulators = []
        for factory in self._accumulator_factories:
            accumulators.append(factory.create(values))

        return pipeline_dp.accumulator.CompoundAccumulator(accumulators)
