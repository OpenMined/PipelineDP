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
import dataclasses
import enum
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

from pipeline_dp import dp_engine as dpeng
from pipeline_dp import aggregate_params
from pipeline_dp import budget_accounting as ba
from pipeline_dp import data_extractors as extractors
from pipeline_dp import pipeline_backend


class GroupsBalance(enum.Enum):
    UNKNOWN = "unknown"
    # Add other types if needed


@dataclasses.dataclass
class ColumnNames:
    """Specifies column names."""
    names: List[str]

    def __init__(self, *names: str):
        self.names = list(names)


@dataclasses.dataclass
class ValuesList:
    """Specifies a list of values."""
    values: Optional[List[Any]]

    def __init__(self, *values: Any):
        self.values = list(values)


class ContributionBoundingLevel(abc.ABC):
    """Abstract class for contribution bounding level."""
    pass


@dataclasses.dataclass
class PartitionLevel(ContributionBoundingLevel):
    """Specifies partition level contribution bounding."""
    max_partitions_contributed: int
    max_contributions_per_partition: int


@dataclasses.dataclass
class RecordLevel(ContributionBoundingLevel):
    """Specifies record level contribution bounding."""
    max_partitions_contributed: int
    # max_contributions_per_partition is implicitly 1


class Budget(abc.ABC):
    """Abstract class for budget."""
    pass


@dataclasses.dataclass
class EpsBudget(Budget):
    """Specifies epsilon budget."""
    epsilon: float


@dataclasses.dataclass
class EpsDeltaBudget(Budget):
    """Specifies epsilon-delta budget."""
    epsilon: float
    delta: float


@dataclasses.dataclass
class StddevBudget(Budget):
    """Specifies standard deviation budget."""
    stddev: float


class GroupBySpec(abc.ABC):
    """Abstract class for specifying grouping."""
    pass


@dataclasses.dataclass
class OptimalGroupSelectionGroupBySpec(GroupBySpec):
    """Specifies optimal group selection."""
    privacy_unit: Optional[ColumnNames]
    default_values_to_ignore: Optional[ValuesList]
    budget: Optional[Budget]
    contribution_bounding_level: ContributionBoundingLevel
    min_privacy_units_per_group: Optional[int]
    public_groups: Optional[Sequence[Any]]
    groups_balance: Optional[GroupsBalance]

    class Builder:

        def __init__(self):
            self._privacy_unit = None
            self._default_values_to_ignore = None
            self._budget = None
            self._contribution_bounding_level = None
            self._min_privacy_units_per_group = None
            self._public_groups = None
            self._groups_balance = None

        def setPrivacyUnit(self,
                           columnNames: ColumnNames,
                           defaultValuesToIgnore: Optional[ValuesList] = None):
            self._privacy_unit = columnNames
            self._default_values_to_ignore = defaultValuesToIgnore
            return self

        def setContributionBoundingLevel(
                self, contributionBoundingLevel: ContributionBoundingLevel):
            self._contribution_bounding_level = contributionBoundingLevel
            return self

        def setBudget(self,
                      eps: Optional[float] = None,
                      delta: Optional[float] = None,
                      stddev: Optional[float] = None):
            if stddev is not None:
                self._budget = StddevBudget(stddev)
            elif delta is not None:
                self._budget = EpsDeltaBudget(eps, delta)
            elif eps is not None:
                self._budget = EpsBudget(eps)
            return self

        def setMinPrivacyUnitsPerGroup(self, minPrivacyUnits: int):
            self._min_privacy_units_per_group = minPrivacyUnits
            return self

        def setPublicGroups(self, publicGroupKeys: Sequence[Any]):
            self._public_groups = publicGroupKeys
            return self

        def setGroupsBalance(self, groupsBalance: GroupsBalance):
            self._groups_balance = groupsBalance
            return self

        def build(self):
            return OptimalGroupSelectionGroupBySpec(
                privacy_unit=self._privacy_unit,
                default_values_to_ignore=self._default_values_to_ignore,
                budget=self._budget,
                contribution_bounding_level=self._contribution_bounding_level,
                min_privacy_units_per_group=self._min_privacy_units_per_group,
                public_groups=self._public_groups,
                groups_balance=self._groups_balance)


class CountSpec(abc.ABC):
    pass


@dataclasses.dataclass
class LaplaceCountSpec(CountSpec):
    privacy_unit: Optional[ColumnNames]
    default_values_to_ignore: Optional[ValuesList]
    contribution_bounding_level: Optional[ContributionBoundingLevel]
    budget: EpsBudget

    class Builder:

        def __init__(self):
            self._privacy_unit = None
            self._default_values_to_ignore = None
            self._contribution_bounding_level = None
            self._budget = None

        def setPrivacyUnit(self,
                           columnNames: ColumnNames,
                           defaultValuesToIgnore: Optional[ValuesList] = None):
            self._privacy_unit = columnNames
            self._default_values_to_ignore = defaultValuesToIgnore
            return self

        def setContributionBoundingLevel(
                self, contributionBoundingLevel: ContributionBoundingLevel):
            self._contribution_bounding_level = contributionBoundingLevel
            return self

        def setBudget(self, eps: float):
            self._budget = EpsBudget(eps)
            return self

        def build(self):
            return LaplaceCountSpec(
                privacy_unit=self._privacy_unit,
                default_values_to_ignore=self._default_values_to_ignore,
                contribution_bounding_level=self._contribution_bounding_level,
                budget=self._budget)


@dataclasses.dataclass
class GaussianCountSpec(CountSpec):
    privacy_unit: Optional[ColumnNames]
    default_values_to_ignore: Optional[ValuesList]
    contribution_bounding_level: Optional[ContributionBoundingLevel]
    budget: EpsDeltaBudget

    class Builder:

        def __init__(self):
            self._privacy_unit = None
            self._default_values_to_ignore = None
            self._contribution_bounding_level = None
            self._budget = None

        def setPrivacyUnit(self,
                           columnNames: ColumnNames,
                           defaultValuesToIgnore: Optional[ValuesList] = None):
            self._privacy_unit = columnNames
            self._default_values_to_ignore = defaultValuesToIgnore
            return self

        def setContributionBoundingLevel(
                self, contributionBoundingLevel: ContributionBoundingLevel):
            self._contribution_bounding_level = contributionBoundingLevel
            return self

        def setBudget(self, eps: float, delta: float):
            self._budget = EpsDeltaBudget(eps, delta)
            return self

        def build(self):
            return GaussianCountSpec(
                privacy_unit=self._privacy_unit,
                default_values_to_ignore=self._default_values_to_ignore,
                contribution_bounding_level=self._contribution_bounding_level,
                budget=self._budget)


class Query:
    """Represents a DP query."""

    def __init__(self, data, group_by_key: ColumnNames,
                 group_by_spec: GroupBySpec, aggregations):
        self._data = data
        self._group_by_key = group_by_key
        self._group_by_spec = group_by_spec
        self._aggregations = aggregations

    def run(self, test_mode: bool = False):
        """Runs the DP query."""
        backend = pipeline_backend.LocalBackend()

        # Initialize Budget Accountant
        total_epsilon = 0
        total_delta = 0

        # Collect budgets
        if self._group_by_spec.budget:
            if isinstance(self._group_by_spec.budget, EpsBudget):
                total_epsilon += self._group_by_spec.budget.epsilon
            elif isinstance(self._group_by_spec.budget, EpsDeltaBudget):
                total_epsilon += self._group_by_spec.budget.epsilon
                total_delta += self._group_by_spec.budget.delta
            # Note: StddevBudget is ignored here for total budget calculation
            # as it implies noise parameters directly.

        for _, _, spec, _ in self._aggregations:
            if hasattr(spec, 'budget') and spec.budget:
                if isinstance(spec.budget, EpsBudget):
                    total_epsilon += spec.budget.epsilon
                elif isinstance(spec.budget, EpsDeltaBudget):
                    total_epsilon += spec.budget.epsilon
                    total_delta += spec.budget.delta

        budget_accountant = ba.NaiveBudgetAccountant(
            total_epsilon=total_epsilon,
            total_delta=total_delta if total_delta > 0 else 1e-10)

        dp_engine = dpeng.DPEngine(budget_accountant=budget_accountant,
                                   backend=backend)

        # 1. Handle Grouping / Partition Selection
        public_partitions = None
        if isinstance(self._group_by_spec, OptimalGroupSelectionGroupBySpec):
            if self._group_by_spec.public_groups is not None:
                public_partitions = self._group_by_spec.public_groups
            elif self._group_by_spec.budget is not None:
                # Perform private partition selection

                group_keys = self._group_by_key.names
                privacy_unit_cols = self._group_by_spec.privacy_unit.names if self._group_by_spec.privacy_unit else []

                def partition_extractor(row):
                    if len(group_keys) == 1:
                        return row[group_keys[0]]
                    return tuple(row[col] for col in group_keys)

                def privacy_id_extractor(row):
                    if privacy_unit_cols:
                        return tuple(row[col] for col in privacy_unit_cols)
                    return id(row)

                max_partitions = 1
                if self._group_by_spec.contribution_bounding_level:
                    if isinstance(
                            self._group_by_spec.contribution_bounding_level,
                            PartitionLevel):
                        max_partitions = self._group_by_spec.contribution_bounding_level.max_partitions_contributed
                    elif isinstance(
                            self._group_by_spec.contribution_bounding_level,
                            RecordLevel):
                        max_partitions = self._group_by_spec.contribution_bounding_level.max_partitions_contributed

                select_params = aggregate_params.SelectPartitionsParams(
                    max_partitions_contributed=max_partitions,
                    partition_selection_strategy=aggregate_params.
                    PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
                    pre_threshold=self._group_by_spec.
                    min_privacy_units_per_group)

                if isinstance(self._group_by_spec.budget, EpsBudget):
                    select_params.budget_weight = self._group_by_spec.budget.epsilon
                elif isinstance(self._group_by_spec.budget, EpsDeltaBudget):
                    select_params.budget_weight = self._group_by_spec.budget.epsilon

                data_extractors = extractors.DataExtractors(
                    partition_extractor=partition_extractor,
                    privacy_id_extractor=privacy_id_extractor,
                    value_extractor=lambda x: 0)

                # Filter data if default_values_to_ignore is set
                select_data = self._data
                if self._group_by_spec.default_values_to_ignore and self._group_by_spec.default_values_to_ignore.values:
                    ignore_vals = set(
                        self._group_by_spec.default_values_to_ignore.values)

                    def filter_ignore(row):
                        # Check if privacy unit columns match any ignored value
                        # Assuming ignored values match the tuple/scalar structure of privacy unit
                        p_id = privacy_id_extractor(row)
                        return p_id not in ignore_vals

                    select_data = backend.filter(select_data, filter_ignore,
                                                 "Filter ignored values")

                public_partitions = dp_engine.select_partitions(
                    select_data, select_params, data_extractors)
                # Do NOT materialize list(public_partitions) here.

        # 2. Run Aggregations (Lazy)
        lazy_aggregations = []

        for agg_type, output_name, spec, extra_params in self._aggregations:

            metrics = []
            noise_kind = aggregate_params.NoiseKind.LAPLACE

            if "Laplace" in spec.__class__.__name__:
                noise_kind = aggregate_params.NoiseKind.LAPLACE
            elif "Gaussian" in spec.__class__.__name__:
                noise_kind = aggregate_params.NoiseKind.GAUSSIAN

            budget_weight = 1.0
            if hasattr(spec, 'budget'):
                if hasattr(spec.budget, 'epsilon'):
                    budget_weight = spec.budget.epsilon

            max_partitions = 1
            max_contributions_per_partition = 1
            if hasattr(spec, 'contribution_bounding_level'
                      ) and spec.contribution_bounding_level:
                if isinstance(spec.contribution_bounding_level, PartitionLevel):
                    max_partitions = spec.contribution_bounding_level.max_partitions_contributed
                    max_contributions_per_partition = spec.contribution_bounding_level.max_contributions_per_partition
                elif isinstance(spec.contribution_bounding_level, RecordLevel):
                    max_partitions = spec.contribution_bounding_level.max_partitions_contributed
                    max_contributions_per_partition = 1

            if agg_type == "count":
                metrics = [aggregate_params.Metrics.COUNT]

            params = aggregate_params.AggregateParams(
                noise_kind=noise_kind,
                metrics=metrics,
                max_partitions_contributed=max_partitions,
                max_contributions_per_partition=max_contributions_per_partition,
                budget_weight=budget_weight,
            )

            group_keys = self._group_by_key.names
            privacy_unit_cols = []
            if hasattr(spec, 'privacy_unit') and spec.privacy_unit:
                privacy_unit_cols = spec.privacy_unit.names

            def partition_extractor(row):
                if len(group_keys) == 1:
                    return row[group_keys[0]]
                return tuple(row[col] for col in group_keys)

            def privacy_id_extractor(row):
                if privacy_unit_cols:
                    return tuple(row[col] for col in privacy_unit_cols)
                return id(row)

            def value_extractor(row):
                return 0

            data_extractors = extractors.DataExtractors(
                partition_extractor=partition_extractor,
                privacy_id_extractor=privacy_id_extractor,
                value_extractor=value_extractor)

            # Filter data if default_values_to_ignore is set
            agg_data = self._data
            if hasattr(
                    spec, 'default_values_to_ignore'
            ) and spec.default_values_to_ignore and spec.default_values_to_ignore.values:
                ignore_vals = set(spec.default_values_to_ignore.values)

                def filter_ignore(row):
                    p_id = privacy_id_extractor(row)
                    return p_id not in ignore_vals

                agg_data = backend.filter(agg_data, filter_ignore,
                                          "Filter ignored values")

            result = dp_engine.aggregate(col=agg_data,
                                         params=params,
                                         data_extractors=data_extractors,
                                         public_partitions=public_partitions)
            lazy_aggregations.append((output_name, result, agg_type))

        budget_accountant.compute_budgets()

        # 3. Materialize Results
        aggregations_results = []
        for output_name, result, agg_type in lazy_aggregations:
            agg_dict = dict(list(result))
            aggregations_results.append((output_name, agg_dict, agg_type))

        all_keys = set()
        for _, res_dict, _ in aggregations_results:
            all_keys.update(res_dict.keys())

        final_results = []
        for key in all_keys:
            row = {}
            if len(self._group_by_key.names) == 1:
                row[self._group_by_key.names[0]] = key
            else:
                for i, col_name in enumerate(self._group_by_key.names):
                    row[col_name] = key[i]

            for output_name, res_dict, agg_type in aggregations_results:
                if key in res_dict:
                    val_dict = res_dict[key]
                    if hasattr(val_dict, '_asdict'):
                        val_dict = val_dict._asdict()

                    if agg_type == "count":
                        row[output_name] = val_dict['count']
                else:
                    row[output_name] = None

            final_results.append(row)

        return final_results


class AggregationBuilder:

    def __init__(self, data, group_by_key, group_by_spec):
        self._data = data
        self._group_by_key = group_by_key
        self._group_by_spec = group_by_spec
        self._aggregations = []

    def count(self, output_column_name: str,
              spec: CountSpec) -> 'AggregationBuilder':
        """Schedules the count aggregation."""
        self._aggregations.append(("count", output_column_name, spec, {}))
        return self

    def build(self) -> Query:
        """Builds the query."""
        return Query(self._data, self._group_by_key, self._group_by_spec,
                     self._aggregations)


class GroupByBuilder:

    def __init__(self, data):
        self._data = data

    def group_by(self, group_keys: ColumnNames,
                 spec: GroupBySpec) -> 'AggregationBuilder':
        """Specifies how to group the data."""
        return AggregationBuilder(self._data, group_keys, spec)


class QueryBuilder:
    """Builds DP queries."""

    def __init__(self):
        self._data = None

    def from_(self, data, *args) -> 'GroupByBuilder':
        """Specifies the data to be processed."""
        self._data = data
        return GroupByBuilder(data)
