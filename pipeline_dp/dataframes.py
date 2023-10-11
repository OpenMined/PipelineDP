import abc
from dataclasses import dataclass

import pandas as pd

import pipeline_dp
from typing import Any, Sequence, Callable, Optional, List, Dict
import pyspark


@dataclass
class _Columns:
    privacy_key: str
    partition_key: str
    value: Optional[str]


@dataclass
class _ContributionBounds:
    max_partitions_contributed: Optional[int] = None
    max_contributions_per_partition: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class DataFrameConvertor(abc.ABC):

    @abc.abstractmethod
    def dataframe_to_collection(df, columns: _Columns):
        pass

    @abc.abstractmethod
    def collection_to_dataframe(col, partition_key_column: str):
        pass


class PandasConverter(DataFrameConvertor):

    def dataframe_to_collection(self, df: pd.DataFrame,
                                columns: _Columns) -> list:
        assert isinstance(df,
                          pd.DataFrame), "Only Pandas dataframes are supported"
        columns_to_keep = [columns.privacy_key, columns.partition_key]
        if columns.value is not None:
            columns_to_keep.append(columns.value)
        df = df[columns_to_keep]  # leave only needed columns.
        if columns.value is None:
            # For count value is not needed, but for simplicity always provide
            # value.
            df['value'] = 0

        # name=None makes that tuples instead of name tuple are returned.
        return list(df.itertuples(index=False, name=None))

    def collection_to_dataframe(self, col: list,
                                partition_key_column: str) -> pd.DataFrame:
        assert isinstance(col, list), "Only local run is supported for now"
        partition_keys, data = list(zip(*col))
        df = pd.DataFrame(data=data)
        df[partition_key_column] = partition_keys
        columns = list(df.columns)
        columns = [columns[-1]] + columns[:-1]
        df = df.reindex(columns=columns).set_index(partition_key_column)
        return df


class SparkConverter(DataFrameConvertor):

    def dataframe_to_collection(self, df, columns: _Columns) -> pyspark.RDD:
        columns_to_keep = [columns.privacy_key, columns.partition_key]
        if columns.value is not None:
            columns_to_keep.append(columns.value)
        df = df[columns_to_keep]  # leave only needed columns.
        return []

    def collection_to_dataframe(self, col: pyspark.RDD,
                                partition_key_column: str):
        pass


def create_backend_for_dataframe(
        df) -> pipeline_dp.pipeline_backend.PipelineBackend:
    if isinstance(df, pd.DataFrame):
        return pipeline_dp.LocalBackend()
    if isinstance(df, pyspark.DataFrame):
        return pipeline_dp.SparkRDDBackend()
    raise NotImplementedError(
        f"Dataframes of type {type(df)} not yet supported")


def create_dataframe_converter(df) -> DataFrameConvertor:
    if isinstance(df, pd.DataFrame):
        return PandasConverter()
    if isinstance(df, pyspark.DataFrame):
        return SparkConverter()
    raise NotImplementedError(
        f"Dataframes of type {type(df)} not yet supported")


@dataclass
class Budget:
    epsilon: float
    delta: float = 0

    # TODO: validate budget.


class Query:

    def __init__(self, df, columns: _Columns, metrics: Dict[pipeline_dp.Metric,
                                                            List[str]],
                 contribution_bonds: _ContributionBounds, public_keys):
        self._df = df
        self._columns = columns
        self._metrics = metrics
        self._contribution_bonds = contribution_bonds
        self._public_partitions = public_keys
        self._expain_computation_report = None

    def run_query(self,
                  budget: Budget,
                  noise_kind: Optional[pipeline_dp.NoiseKind] = None):
        converter = create_dataframe_converter(self._df)
        col = converter.dataframe_to_collection(self._df, self._columns)
        backend = create_backend_for_dataframe(self._df)
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(
            total_epsilon=budget.epsilon, total_delta=budget.delta)

        dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)
        params = pipeline_dp.AggregateParams(
            noise_kind=noise_kind,
            metrics=self._metrics,
            max_partitions_contributed=self._contribution_bonds.
            max_partitions_contributed,
            max_contributions_per_partition=self._contribution_bonds.
            max_contributions_per_partition,
            min_value=self._contribution_bonds.min_value,
            max_value=self._contribution_bonds.max_value)

        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda row: row[0],
            partition_extractor=lambda row: row[1],
            value_extractor=lambda row: row[2])

        explain_computation_report = pipeline_dp.ExplainComputationReport()

        dp_result = dp_engine.aggregate(
            col,
            params,
            data_extractors,
            public_partitions=self._public_partitions,
            out_explain_computation_report=explain_computation_report)
        budget_accountant.compute_budgets()
        dp_result = list(dp_result)
        self._expain_computation_report = explain_computation_report.text()
        return converter.collection_to_dataframe(dp_result,
                                                 self._columns.partition_key)

    def explain_computations(self):
        if self._expain_computation_report is None:
            raise ValueError("Query is not run yet. Call run_query first")
        return self._expain_computation_report


class QueryBuilder:

    def __init__(self, df, privacy_key_column: str):
        self._df = df
        self._privacy_key_column = privacy_key_column  # todo: check df.scheme
        self._groupby_column = None
        self._value_column = None
        self._metrics = {}
        self._contribution_bounds = _ContributionBounds()
        self._public_keys = None

    def groupby(self,
                column: str,
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
                "Global aggregations are not implemented yet.")
        metrics = list(self._metrics.keys())
        return Query(
            self._df,
            _Columns(self._privacy_key_column, self._groupby_column,
                     self._value_column), metrics, self._contribution_bounds,
            self._public_keys)
