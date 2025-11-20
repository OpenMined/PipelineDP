# Copyright 2023 OpenMined.
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
"""Advanced query builder API."""

import abc
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Union
from collections import defaultdict, namedtuple

from pipeline_dp import aggregate_params, pipeline_backend, DPEngine, NaiveBudgetAccountant, DataExtractors
from pipeline_dp.pipeline_backend import LocalBackend


@dataclass
class ColumnNames:
    """Specifies column names."""
    names: Union[str, Sequence[str]]


@dataclass
class EpsDeltaBudget:
    """Specifies epsilon and delta for a DP operation."""
    epsilon: float
    delta: float = 0.0


@dataclass
class EpsBudget:
    """Specifies epsilon for a DP operation."""
    epsilon: float


@dataclass
class StddevBudget:
    """Specifies standard deviation for a DP operation."""
    stddev: float


Budget = Union[EpsDeltaBudget, EpsBudget, StddevBudget]


@dataclass
class ContributionBoundingLevel:
    """Specifies contribution bounding level."""
    max_partitions_contributed: int
    max_contributions_per_partition: int


class GroupBySpec(abc.ABC):
    """Abstract class for specifying grouping."""
    pass


@dataclass
class OptimalGroupSelectionGroupBySpec(GroupBySpec):
    """Specifies optimal group selection."""
    privacy_unit: ColumnNames
    budget: Budget
    contribution_bounding_level: ContributionBoundingLevel
    min_privacy_units_per_group: Optional[int] = None
    public_groups: Optional[Sequence[Any]] = None
    pre_threshold: Optional[int] = None


class CountSpec(abc.ABC):
    """Abstract class for specifying count."""
    budget: Budget
    privacy_unit: Optional[ColumnNames] = None
    contribution_bounding_level: Optional[ContributionBoundingLevel] = None


@dataclass
class LaplaceCountSpec(CountSpec):
    """Specifies count with the Laplace mechanism."""
    budget: EpsBudget


@dataclass
class GaussianCountSpec(CountSpec):
    """Specifies count with the Gaussian mechanism."""
    budget: EpsDeltaBudget


class Query:
    """Represents a DP query."""

    def __init__(self, data, group_by_spec, aggregations):
        self._data = data
        self._group_by_spec = group_by_spec
        self._aggregations = aggregations

    def run(self):
        """Runs the DP query."""
        backend = LocalBackend()

        # Calculate total budget
        total_epsilon, total_delta = 0, 0
        for agg in self._aggregations:
            spec = agg[-1]
            if hasattr(spec.budget, 'epsilon'):
                total_epsilon += spec.budget.epsilon
            if hasattr(spec.budget, 'delta'):
                total_delta += spec.budget.delta
        if self._group_by_spec:
            spec = self._group_by_spec[1]
            if hasattr(spec.budget, 'epsilon'):
                total_epsilon += spec.budget.epsilon
            if hasattr(spec.budget, 'delta'):
                total_delta += spec.budget.delta

        budget_accountant = NaiveBudgetAccountant(total_epsilon=total_epsilon,
                                                  total_delta=total_delta if total_delta > 0 else 1e-10)
        dp_engine = DPEngine(budget_accountant=budget_accountant, backend=backend)

        metrics = []
        for agg_type, *agg_args in self._aggregations:
            if agg_type == "count":
                metrics.append(aggregate_params.Metrics.COUNT)

        params = aggregate_params.AggregateParams(
            noise_kind=aggregate_params.NoiseKind.LAPLACE,
            metrics=metrics,
            max_partitions_contributed=self._group_by_spec[1].contribution_bounding_level.max_partitions_contributed,
            max_contributions_per_partition=self._group_by_spec[
                1].contribution_bounding_level.max_contributions_per_partition,
            pre_threshold=self._group_by_spec[1].pre_threshold
        )

        def _get_extractor(columns: Union[str, Sequence[str]]) -> callable:
            if isinstance(columns, str):
                return lambda row: row[columns]
            else:
                return lambda row: tuple(row[col] for col in columns)

        data_extractors = DataExtractors(
            partition_extractor=_get_extractor(self._group_by_spec[0].names),
            privacy_id_extractor=_get_extractor(self._group_by_spec[1].privacy_unit.names),
            value_extractor=lambda row: 0
        )

        dp_result = dp_engine.aggregate(
            col=self._data,
            params=params,
            data_extractors=data_extractors,
            public_partitions=self._group_by_spec[1].public_groups
        )

        budget_accountant.compute_budgets()

        return list(dp_result)


class QueryBuilder:
    """Builds DP queries."""

    def __init__(self):
        self._data = None
        self._group_by_spec = None
        self._aggregations = []

    def from_(self, data) -> 'QueryBuilder':
        self._data = data
        return self

    def group_by(self, group_keys: ColumnNames, spec: GroupBySpec) -> 'QueryBuilder':
        self._group_by_spec = (group_keys, spec)
        return self

    def count(self, output_column_name: str, spec: CountSpec) -> 'QueryBuilder':
        self._aggregations.append(("count", output_column_name, spec))
        return self

    def build(self) -> Query:
        return Query(self._data, self._group_by_spec, self._aggregations)
