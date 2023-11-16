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
import abc
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import pipeline_dp
import pyspark

SparkDataFrame = pyspark.sql.dataframe.DataFrame


@dataclass
class Budget:
    epsilon: float
    delta: float = 0

    def __post_init__(self):
        pipeline_dp.input_validators.validate_epsilon_delta(
            self.epsilon, self.delta, "Budget")


@dataclass
class Columns:
    privacy_key: str
    partition_key: Union[str, Sequence[str]]
    value: Optional[str]


@dataclass
class ContributionBounds:
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class DataFrameConvertor(abc.ABC):
    """Base class for conversion between DataFrames and Collections."""

    @abc.abstractmethod
    def dataframe_to_collection(df, columns: Columns):
        pass

    @abc.abstractmethod
    def collection_to_dataframe(col, group_key_column: str):
        pass


class SparkConverter(DataFrameConvertor):
    """Convertor between RDD and Spark DataFrame."""

    def __init__(self, spark: pyspark.sql.SparkSession):
        self._spark = spark

    def dataframe_to_collection(self, df: SparkDataFrame,
                                columns: Columns) -> pyspark.RDD:
        columns_to_keep = [columns.privacy_key]
        if isinstance(columns.partition_key, str):
            num_partition_columns = 1
            columns_to_keep.append(columns.partition_key)
        else:  # Sequence[str], multiple columns
            num_partition_columns = len(columns.partition_key)
            columns_to_keep.extend(columns.partition_key)
        value_present = columns.value is not None
        if value_present:
            columns_to_keep.append(columns.value)

        df = df[columns_to_keep]  # leave only needed columns.

        def extractor(row):
            privacy_key = row[0]
            partition_key = row[1] if num_partition_columns == 1 else row[
                1:1 + num_partition_columns]
            value = row[1 + num_partition_columns] if value_present else 0
            return (privacy_key, partition_key, value)

        return df.rdd.map(extractor)

    def collection_to_dataframe(self, col: pyspark.RDD) -> SparkDataFrame:
        return self._spark.createDataFrame(col)


def _create_backend_for_dataframe(
        df: SparkDataFrame) -> pipeline_dp.PipelineBackend:
    """Creates a pipeline backend based on type of DataFrame."""
    if isinstance(df, SparkDataFrame):
        return pipeline_dp.SparkRDDBackend(df.sparkSession.sparkContext)
    raise NotImplementedError(
        f"Dataframes of type {type(df)} not yet supported")


def _create_dataframe_converter(df: SparkDataFrame) -> DataFrameConvertor:
    """Creates a DataConvert based on type of DataFrame."""
    if isinstance(df, SparkDataFrame):
        return SparkConverter(df.sparkSession)
    raise NotImplementedError(
        f"Dataframes of type {type(df)} not yet supported")


class Query:
    """DP Query on DataFrames.

    Warning: Don't create it directly, use QueryBuilder().
    """

    def __init__(self, df: SparkDataFrame, columns: Columns,
                 metrics_output_columns: Dict[pipeline_dp.Metric, str],
                 contribution_bounds: ContributionBounds,
                 public_partitions: Optional[Iterable]):
        """Constructor.

        Warning: Don't create it directly, use QueryBuilder().

        Args:
            df: DataFrame with data to be anonymized.
            columns: privacy_key, partition and value columns in df.
            metrics_output_columns: mapping from metrics to the output DataFrame
              column.
            contribution_bounds: contribution bounds for computing DP
              aggregation.
            public_partitions: public partitions, in case if they are known.
        """
        self._df = df
        self._columns = columns
        self._metrics_output_columns = metrics_output_columns
        self._contribution_bounds = contribution_bounds
        self._public_partitions = public_partitions

    def run_query(
            self,
            budget: Budget,
            noise_kind: pipeline_dp.NoiseKind = pipeline_dp.NoiseKind.LAPLACE):
        """Runs the DP query and returns a DataFrame.

        Args:
             budget: DP budget
             noise_kind: type of noise which is used for the anonymization.
        Returns:
            DataFrame with DP aggregation result.
        """
        converter = _create_dataframe_converter(self._df)
        backend = _create_backend_for_dataframe(self._df)
        col = converter.dataframe_to_collection(self._df, self._columns)
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(
            total_epsilon=budget.epsilon, total_delta=budget.delta)

        dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)
        metrics = list(self._metrics_output_columns.keys())
        params = pipeline_dp.AggregateParams(
            noise_kind=noise_kind,
            metrics=metrics,
            max_partitions_contributed=self._contribution_bounds.
            max_partitions_contributed,
            max_contributions_per_partition=self._contribution_bounds.
            max_contributions_per_partition,
            min_value=self._contribution_bounds.min_value,
            max_value=self._contribution_bounds.max_value)

        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda row: row[0],
            partition_extractor=lambda row: row[1],
            value_extractor=lambda row: row[2])

        dp_result = dp_engine.aggregate(
            col,
            params,
            data_extractors,
            public_partitions=self._public_partitions)
        # dp_result: (partition_key, NamedTuple(metrics))
        budget_accountant.compute_budgets()

        # Convert elements to named tuple.
        metrics_names_to_output_columns = {}
        for metric, output_column in self._metrics_output_columns.items():
            metric_name = metric.name.lower()
            if output_column is None:
                output_column = metric_name  # default name
            metrics_names_to_output_columns[metric_name] = output_column

        output_columns = list(metrics_names_to_output_columns.values())
        partition_key = self._columns.partition_key
        partition_key_one_column = isinstance(partition_key, str)
        if partition_key_one_column:
            partition_key = [partition_key]
        PartitionMetricsTuple = namedtuple("Result",
                                           partition_key + output_columns)

        def convert_to_partition_metrics_tuple(row):
            partition, metrics = row
            if partition_key_one_column:
                result = {partition_key[0]: partition}
            else:
                result = {}
                for key, value in zip(partition_key, partition):
                    result[key] = value
            for key, value in metrics._asdict().items():
                # Map default metric names to metric names specified in
                # self.metrics_names_to_output_columns
                result[metrics_names_to_output_columns[key]] = value
            return PartitionMetricsTuple(**result)

        dp_result = backend.map(dp_result, convert_to_partition_metrics_tuple,
                                "Convert to NamedTuple")
        # dp_result: PartitionMetricsTuple

        return converter.collection_to_dataframe(dp_result)


@dataclass
class _AggregationSpec:
    metric: pipeline_dp.Metric
    input_column: Optional[str]
    output_column: Optional[str]
    min_value: Optional[float] = None
    max_value: Optional[float] = None


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
        .mean("money_spent") # no need to specify min_value, max_value, they
        # were already given in sum
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
        self._by = None
        self._public_keys = None
        self._aggregations_specs = []
        self._max_partitions_contributed: int = None
        self._max_contributions_per_partition: int = None

    def groupby(
            self,
            by: Union[str, Sequence[str]],
            *,
            max_groups_contributed: int,
            max_contributions_per_group: int,
            public_keys: Optional[Iterable[Any]] = None,
            column: Optional[str] = None  # deprecated
    ) -> 'QueryBuilder':
        """Adds groupby by the given column to the query.

        All following aggregation will be applied to grouped by DataFrame.

        Args:
            by: a column or a list of columns used to determine the groups.
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
        if column is not None:
            raise ValueError("column argument is deprecated. Use by")
        if self._by is not None:
            raise ValueError("groupby can be called only once")

        if isinstance(by, str):
            if by not in self._df.columns:
                raise ValueError(f"Column {by} is not present in DataFrame")
        elif isinstance(by, list):
            # List of columns
            for column in by:
                if column not in self._df.columns:
                    raise ValueError(
                        f"Column {column} is not present in DataFrame")
        else:
            raise ValueError(f"by argument must be column name(s)")
        self._by = by
        self._max_partitions_contributed = max_groups_contributed
        self._max_contributions_per_partition = max_contributions_per_group
        self._public_keys = public_keys
        return self

    def count(self, name: str = None) -> 'QueryBuilder':
        """Adds count to the query.

        Args:
            name: the name of the output column.
        """
        return self._add_aggregation(
            _AggregationSpec(metric=pipeline_dp.Metrics.COUNT,
                             input_column=None,
                             output_column=name))

    def sum(self,
            column: str,
            *,
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            name: str = None) -> 'QueryBuilder':
        """Adds sum aggregation to the query.

        Args:
            column: column to perform summation
            min_value, max_value: capping limits to each value.
            name: the name of the output column.
        """
        return self._add_aggregation(
            _AggregationSpec(metric=pipeline_dp.Metrics.SUM,
                             input_column=column,
                             output_column=name,
                             min_value=min_value,
                             max_value=max_value))

    def mean(self,
             column: str,
             *,
             min_value: Optional[float] = None,
             max_value: Optional[float] = None,
             name: str = None) -> 'QueryBuilder':
        """Adds mean aggregation to the query.

        Args:
            column: column to compute mean
            min_value, max_value: capping limits to each value.
            name: the name of the output column.
        """

        return self._add_aggregation(
            _AggregationSpec(metric=pipeline_dp.Metrics.MEAN,
                             input_column=column,
                             output_column=name,
                             min_value=min_value,
                             max_value=max_value))

    def build_query(self) -> Query:
        """Builds the DP query."""
        if self._by is None:
            raise NotImplementedError(
                "Global aggregations are not implemented yet. Call groupby")
        # Validation.
        if not self._aggregations_specs:
            raise ValueError(
                "No aggregations in the query. Call for example count")
        # 1. Not more than 1 value column
        input_columns = [
            spec.input_column
            for spec in self._aggregations_specs
            if spec.input_column is not None
        ]
        if len(set(input_columns)) > 1:
            raise NotImplementedError(
                f"Aggregation of only 1 column is supported, but {input_columns} given"
            )
        input_column = input_columns[0] if input_columns else None
        # 2. Each metric only once
        metrics = [spec.metric for spec in self._aggregations_specs]
        if len(set(metrics)) != len(metrics):
            raise ValueError("Each aggregation can be added only once.")
        # 3. Value caps are given if needed and consistent
        min_value, max_value = self._get_value_caps()

        # Create Contribution bounds
        contribution_bounds = ContributionBounds(
            max_partitions_contributed=self._max_partitions_contributed,
            max_contributions_per_partition=self.
            _max_contributions_per_partition,
            min_value=min_value,
            max_value=max_value)

        metric_to_output_column = dict((spec.metric, spec.output_column)
                                       for spec in self._aggregations_specs)

        return Query(self._df,
                     Columns(self._privacy_unit_column, self._by,
                             input_column), metric_to_output_column,
                     contribution_bounds, self._public_keys)

    def _add_aggregation(self,
                         aggregation_spec: _AggregationSpec) -> 'QueryBuilder':
        if self._by is None:
            raise ValueError(
                "Global aggregations are not supported. Use groupby")
        self._aggregations_specs.append(aggregation_spec)
        return self

    def _get_value_caps(self) -> Tuple[Optional[float], Optional[float]]:
        metrics = set([spec.metric for spec in self._aggregations_specs])
        # COUNT, PPRIVACY_ID_COUNT do not require caps.
        metrics_which_need_caps = metrics.difference(
            [pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT])
        if not metrics_which_need_caps:
            return None, None
        min_values = [
            spec.min_value
            for spec in self._aggregations_specs
            if spec.min_value is not None
        ]
        max_values = [
            spec.max_value
            for spec in self._aggregations_specs
            if spec.max_value is not None
        ]
        if not min_values or not max_values:
            raise ValueError("min_value and max_value should be given at least "
                             "once as arguments of sum or mean")

        def all_values_equal(a: list) -> bool:
            return min(a) == max(a)

        if not all_values_equal(min_values) or not all_values_equal(max_values):
            raise ValueError("If min_value and max_value provided multiple "
                             "times they must be the same")
        return min_values[0], max_values[0]
