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
from typing import Optional, List, Dict, Iterable

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

    def __init__(self, df, columns: Columns, metrics: Dict[pipeline_dp.Metric,
                                                           List[str]],
                 contribution_bonds: ContributionBounds, public_keys):
        self._df = df
        self._columns = columns
        self._metrics = metrics
        self._contribution_bonds = contribution_bonds
        self._public_partitions = public_keys
        self._expain_computation_report = None

    def run_query(self,
                  budget: Budget,
                  noise_kind: Optional[pipeline_dp.NoiseKind] = None):
        raise NotImplementedError("Running query is not yet implemented")


class QueryBuilder:

    def __init__(self, df, privacy_key_column: str):
        if privacy_key_column not in df.scheme:
            raise ValueError(
                f"Column {privacy_key_column} is not present in DataFrame")
        self._df = df
        self._privacy_key_column = privacy_key_column
        self._groupby_column = None
        self._value_column = None
        self._metrics = {}  # map from pipeline_dp.Metric -> output column name
        self._contribution_bounds = ContributionBounds()
        self._public_keys = None

    def groupby(self,
                column: str,
                *,
                max_partitions_contributed: int,
                max_contributions_per_partition: int,
                public_keys=None) -> 'QueryBuilder':
        if self._groupby_column is not None:
            raise ValueError("groupby can be called only once.")
        self._groupby_column = column
        self._contribution_bounds.max_partitions_contributed = max_partitions_contributed
        self._contribution_bounds.max_contributions_per_partition = max_contributions_per_partition
        self._public_keys = public_keys
        return self

    def count(self, name: str = None) -> 'QueryBuilder':
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
        if self._groupby_column is None:
            raise ValueError(
                "Global aggregations are not supported. Use groupby.")
        if pipeline_dp.Metrics.SUM in self._metrics:
            raise ValueError("sum can be counted only once.")
        self._metrics[pipeline_dp.Metrics.SUM] = name
        self._value_column = column
        self._contribution_bounds.min_value = min_value
        self._contribution_bounds.max_value = max_value
        return self

    def build_query(self) -> Query:
        if self._groupby_column is None:
            raise NotImplementedError(
                "Global aggregations are not implemented yet. Call groupby.")
        metrics = list(self._metrics.keys())
        return Query(
            self._df,
            Columns(self._privacy_key_column, self._groupby_column,
                    self._value_column), metrics, self._contribution_bounds,
            self._public_keys)
