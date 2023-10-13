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
"""Computing DP aggregations on (Pandas, Spark, Beam) Dataframes."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pipeline_dp


@dataclass
class Columns:
    privacy_key: str
    partition_key: str
    value: Optional[str]


@dataclass
class ContributionBounds:
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class Budget:
    epsilon: float
    delta: float = 0

    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, not {self.epsilon}.")
        if self.delta < 0:
            raise ValueError(f"delta must be non-negative, not {self.delta}.")


class Query:
    """DP Query on DataFrames.

    Warning: Don't create it directly, use QueryBuilder().
    """

    def __init__(self, df, columns: Columns, metrics: Dict[pipeline_dp.Metric,
                                                           List[str]],
                 contribution_bounds: ContributionBounds, public_keys):
        self._df = df
        self._columns = columns
        self._metrics = metrics
        self._contribution_bounds = contribution_bounds
        self._public_partitions = public_keys

    def run_query(self,
                  budget: Budget,
                  noise_kind: Optional[pipeline_dp.NoiseKind] = None):
        raise NotImplementedError("Running query is not yet implemented")


class QueryBuilder:
    """Builds DP queries on (Spark, Pandas, Beam) DataFrames.

    It uses a Builder pattern, each public function except of build_query
    returns the reference to itself.
    The usage is
    QueryBuilder().groupby(...).aggregation1(...)....aggregation_n().build_query()

    Example:
        # The following query computes restaurant visits and total money
        # spent:
        query = QueryBuilder(df, "user_id")
        .groupby("day", max_groups_contributed=3, max_contributions_per_group=1)
        .count()
        .sum("money_spent", min_value=0, max_value=100)
        .build_query()
    """

    def __init__(self, df, privacy_unit_column: str):
        """Constructor.

        Args:
            df: (Spark, Pandas, Beam) DataFrame with data to anonymize.
            privacy_key_column: column from `df` with privacy key.
        """
        if privacy_unit_column not in df.columns:
            raise ValueError(
                f"Column {privacy_unit_column} is not present in DataFrame")
        self._df = df
        self._privacy_unit_column = privacy_unit_column
        self._groupby_column = None
        self._value_column = None
        self._metrics = {}  # map from pipeline_dp.Metric -> output column name
        self._contribution_bounds = ContributionBounds()
        self._public_keys = None

    def groupby(self,
                column: str,
                *,
                max_groups_contributed: int,
                max_contributions_per_group: int,
                public_keys: Optional[Iterable[Any]] = None) -> 'QueryBuilder':
        """Adds groupby by the given column to the query.

        All following aggregation will be applied to grouped by DataFrame.

        Args:
            column: column to group.
            max_groups_contributed: the maximum groups that can each privacy
              unit contributes to the result. If some privacy unit contributes
              more in the input dataset, the groups are sub-sampled to
              max_groups_contributed.
            max_contributions_per_group: the maximum contribution that a privacy
              unit can contribute to a group. If some privacy unit contributes
              more to some group, contributions are sub-sampled to
              max_contributions_per_group.
            public_keys:
        """
        if self._groupby_column is not None:
            raise ValueError("groupby can be called only once.")
        if column not in self._df.columns:
            raise ValueError(f"Column {column} is not present in DataFrame")
        self._groupby_column = column
        self._contribution_bounds.max_partitions_contributed = max_groups_contributed
        self._contribution_bounds.max_contributions_per_partition = max_contributions_per_group
        self._public_keys = public_keys
        return self

    def count(self, name: str = None) -> 'QueryBuilder':
        """Adds count to the query.

        Args:
            name: the name of the output column.
        """
        if self._groupby_column is None:
            raise ValueError(
                "Global aggregations are not supported. Use groupby.")
        if pipeline_dp.Metrics.COUNT in self._metrics:
            raise ValueError("count can be counted only once.")
        self._metrics[pipeline_dp.Metrics.COUNT] = name
        return self

    def sum(self,
            column: str,
            *,
            min_value: float,
            max_value: float,
            name: str = None) -> 'QueryBuilder':
        """Adds sum to the query.

        Args:
            column: column to perform summation
            min_value, max_value: capping limits to each value.
            name: the name of the output column.
        """
        if self._groupby_column is None:
            raise ValueError(
                "Global aggregations are not supported. Use groupby.")
        if pipeline_dp.Metrics.SUM in self._metrics:
            raise ValueError("sum can be counted only once.")
        if self._value_column is not None:
            raise NotImplementedError(
                "Now aggregation of only 1 column is supported.")
        self._metrics[pipeline_dp.Metrics.SUM] = name
        self._value_column = column
        self._contribution_bounds.min_value = min_value
        self._contribution_bounds.max_value = max_value
        return self

    def build_query(self) -> Query:
        """Builds the DP query."""
        if self._groupby_column is None:
            raise NotImplementedError(
                "Global aggregations are not implemented yet. Call groupby.")
        if not self._metrics:
            raise ValueError(
                "No aggregations in the query. Call for example count.")
        metrics = list(self._metrics.keys())
        return Query(
            self._df,
            Columns(self._privacy_unit_column, self._groupby_column,
                    self._value_column), metrics, self._contribution_bounds,
            self._public_keys)
