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
"""Tests for DP aggregations on DataFrames."""
from absl.testing import absltest
from absl.testing import parameterized
from collections import namedtuple
import pandas as pd
from pyspark import sql
from unittest import mock
from unittest.mock import patch

from pipeline_dp import dataframes
import pipeline_dp


def get_pandas_df() -> pd.DataFrame:
    return pd.DataFrame({
        "privacy_key": [0, 1, 1],
        "group_key": ["key1", "key2", "key1"],
        "value": [5.0, 2.5, -2],
        "value2": [1.0, -2.0, 4]
    })


class QueryBuilderTest(parameterized.TestCase):

    def test_incorrect_privacy_unit_column(self):
        df = get_pandas_df()
        with self.assertRaisesRegex(ValueError,
                                    "Column not_present_column is not present"):
            dataframes.QueryBuilder(df, "not_present_column")

    def test_global_aggregations_are_not_supported(self):
        df = get_pandas_df()
        builder = dataframes.QueryBuilder(df, "privacy_key")
        with self.assertRaisesRegex(NotImplementedError,
                                    "Global aggregations are not implemented"):
            builder.count()

    def test_no_aggregation(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        with self.assertRaisesRegex(ValueError, "No aggregations in the query"):
            builder.groupby("group_key",
                            max_groups_contributed=1,
                            max_contributions_per_group=1).build_query()

    def test_group_by_not_existing_column(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        with self.assertRaisesRegex(
                ValueError,
                "Column not_present_column is not present in DataFrame"):
            builder.groupby("not_present_column",
                            max_groups_contributed=1,
                            max_contributions_per_group=1)

    def test_2_group_by(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        with self.assertRaisesRegex(ValueError,
                                    "groupby can be called only once"):
            builder.groupby("group_key",
                            max_groups_contributed=1,
                            max_contributions_per_group=1)

    def test_no_aggregations(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        with self.assertRaisesRegex(ValueError, "No aggregations in the query"):
            builder.build_query()

    def test_only_one_column_is_supported(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        builder.sum("value").mean("value2")
        with self.assertRaisesRegex(
                NotImplementedError,
                "Aggregation of only one column is supported"):
            builder.build_query()

    def test_same_metrics_added_twice(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        builder.sum("value").sum("value")
        with self.assertRaisesRegex(ValueError,
                                    "Each aggregation can be added only once"):
            builder.build_query()

    def test_caps_not_given(self):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        builder.sum("value")
        with self.assertRaisesRegex(ValueError,
                                    "min_value and max_value must be given"):
            builder.build_query()

    @parameterized.named_parameters(
        dict(testcase_name='min differs', mean_min_value=-1, mean_max_value=1),
        dict(testcase_name='max differs', mean_min_value=0, mean_max_value=2),
        dict(testcase_name='both differ', mean_min_value=5, mean_max_value=10),
    )
    def test_caps_are_different(self, mean_min_value, mean_max_value):
        builder = dataframes.QueryBuilder(get_pandas_df(), "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        builder.sum("value", min_value=0,
                    max_value=1).mean("value",
                                      min_value=mean_min_value,
                                      max_value=mean_max_value)
        with self.assertRaisesRegex(
                ValueError,
                "If min_value and max_value provided multiple times they must be the same"
        ):
            builder.build_query()

    def test_count_query(self):
        df = get_pandas_df()

        query = dataframes.QueryBuilder(df, "privacy_key").groupby(
            "group_key",
            max_groups_contributed=5,
            max_contributions_per_group=10).count().build_query()

        self.assertTrue(query._df.equals(df))
        self.assertEqual(query._columns,
                         dataframes.Columns("privacy_key", "group_key", None))
        self.assertEqual(query._metrics_output_columns,
                         {pipeline_dp.Metrics.COUNT: None})
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=5,
                                          max_contributions_per_partition=10))
        self.assertIsNone(query._public_partitions)

    def test_sum_query(self):
        df = get_pandas_df()

        query = dataframes.QueryBuilder(df, "privacy_key").groupby(
            "group_key",
            max_groups_contributed=8,
            max_contributions_per_group=11).sum("value",
                                                min_value=1,
                                                max_value=2.5).build_query()

        self.assertTrue(query._df.equals(df))
        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key", "group_key", "value"))
        self.assertEqual(query._metrics_output_columns,
                         {pipeline_dp.Metrics.SUM: None})
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=8,
                                          max_contributions_per_partition=11,
                                          min_value=1,
                                          max_value=2.5))

    def test_count_and_sum_query(self):
        query = dataframes.QueryBuilder(get_pandas_df(), "privacy_key").groupby(
            "group_key",
            max_groups_contributed=8,
            max_contributions_per_group=11).count().sum(
                "value", min_value=1, max_value=2.5, name="SUM1").build_query()

        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key", "group_key", "value"))
        self.assertEqual(query._metrics_output_columns, {
            pipeline_dp.Metrics.COUNT: None,
            pipeline_dp.Metrics.SUM: "SUM1"
        })
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=8,
                                          max_contributions_per_partition=11,
                                          min_value=1,
                                          max_value=2.5))

    def test_count_and_mean_query(self):
        query = dataframes.QueryBuilder(get_pandas_df(), "privacy_key").groupby(
            "group_key",
            max_groups_contributed=8,
            max_contributions_per_group=11).count().sum(
                "value", name="SUM1").mean("value", min_value=0,
                                           max_value=3).build_query()

        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key", "group_key", "value"))
        self.assertEqual(
            query._metrics_output_columns, {
                pipeline_dp.Metrics.COUNT: None,
                pipeline_dp.Metrics.SUM: "SUM1",
                pipeline_dp.Metrics.MEAN: None
            })
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=8,
                                          max_contributions_per_partition=11,
                                          min_value=0,
                                          max_value=3))

    def test_public_keys(self):
        query = dataframes.QueryBuilder(get_pandas_df(), "privacy_key").groupby(
            "group_key",
            max_groups_contributed=5,
            max_contributions_per_group=10,
            public_keys=["key1"]).count().build_query()

        self.assertEqual(query._public_partitions, ["key1"])

    def test_partition_key_multiple_columns_query(self):
        df = get_pandas_df()
        df["second_group_by_key"] = ["a", "b", "c"]

        query = dataframes.QueryBuilder(df, "privacy_key").groupby(
            ["group_key", "second_group_by_key"],
            max_groups_contributed=8,
            max_contributions_per_group=11).sum("value",
                                                min_value=1,
                                                max_value=2.5).build_query()

        self.assertTrue(query._df.equals(df))
        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key",
                               ["group_key", "second_group_by_key"], "value"))
        self.assertEqual(query._metrics_output_columns,
                         {pipeline_dp.Metrics.SUM: None})


CountNamedTuple = namedtuple("Count", ["count"])
CountSumNamedTuple = namedtuple("CountSum", ["count", "sum"])
CountMeanNamedTuple = namedtuple("CountMean", ["count", "mean"])


class QueryTest(parameterized.TestCase):

    def _get_spark_session(self) -> sql.SparkSession:
        return sql.SparkSession.builder.appName("Test").getOrCreate()

    @parameterized.named_parameters(
        dict(testcase_name='count, public_partitions',
             metrics=[pipeline_dp.Metrics.COUNT],
             public_keys=["key1"]),
        dict(testcase_name='sum, count, private partitions',
             metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
             public_keys=None),
        dict(testcase_name='mean, count, private partitions',
             metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.MEAN],
             public_keys=None))
    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_run_query(self, mock_aggregate, metrics, public_keys):
        # Arrange
        spark = self._get_spark_session()
        if pipeline_dp.Metrics.SUM in metrics:
            mock_aggregate.return_value = spark.sparkContext.parallelize([
                ("key1", CountSumNamedTuple(1.2345, 2.34567))
            ])
        elif pipeline_dp.Metrics.MEAN in metrics:
            mock_aggregate.return_value = spark.sparkContext.parallelize([
                ("key1", CountMeanNamedTuple(1.2345, 2.34567))
            ])
        else:
            mock_aggregate.return_value = spark.sparkContext.parallelize([
                ("key1", CountNamedTuple(1.2345))
            ])
        df = spark.createDataFrame(get_pandas_df())
        value_column = "value" if pipeline_dp.Metrics.SUM in metrics else None
        columns = dataframes.Columns("privacy_key", "group_key", value_column)
        metrics_dict = dict([(m, m.name) for m in metrics])
        min_value = max_value = None
        if pipeline_dp.Metrics.SUM in metrics or pipeline_dp.Metrics.MEAN in metrics:
            min_value, max_value = -5, 5
        bounds = dataframes.ContributionBounds(
            max_partitions_contributed=5,
            max_contributions_per_partition=2,
            min_value=min_value,
            max_value=max_value)
        query = dataframes.Query(df, columns, metrics_dict, bounds, public_keys)

        # Act
        result = query.run_query(dataframes.Budget(1, 1e-10))

        # Assert
        expected_aggregate_params = pipeline_dp.AggregateParams(
            max_partitions_contributed=5,
            max_contributions_per_partition=2,
            min_value=min_value,
            max_value=max_value,
            metrics=metrics)
        mock_aggregate.assert_called_once_with(mock.ANY,
                                               expected_aggregate_params,
                                               mock.ANY,
                                               public_partitions=public_keys)
        df = result.toPandas()
        self.assertLen(df, 1)
        row = df.loc[0]
        self.assertEqual(row["group_key"], "key1")
        self.assertEqual(row["COUNT"], 1.2345)

    def test_run_query_e2e_run(self):
        # Arrange
        spark = self._get_spark_session()
        df = spark.createDataFrame(get_pandas_df())
        columns = dataframes.Columns("privacy_key", "group_key", "value")
        metrics = {
            pipeline_dp.Metrics.COUNT: "count_column",
            pipeline_dp.Metrics.SUM: None,  # it returns default name "sum"
            pipeline_dp.Metrics.MEAN: None  # it returns default name "mean"
        }
        bounds = dataframes.ContributionBounds(
            max_partitions_contributed=2,
            max_contributions_per_partition=2,
            min_value=-5,
            max_value=5)
        public_keys = ["key1", "key0"]
        query = dataframes.Query(df, columns, metrics, bounds, public_keys)

        # Act
        budget = dataframes.Budget(1e6, 1e-1)  # large budget to get small noise
        result_df = query.run_query(budget)

        # Assert
        pandas_df = result_df.toPandas()
        pandas_df = pandas_df.sort_values(by=['group_key']).reset_index(
            drop=True)
        self.assertLen(pandas_df, 2)
        # check row[0] = "key0", 0+noise, 0+noise
        row0 = pandas_df.loc[0]
        self.assertEqual(row0["group_key"], "key0")
        self.assertAlmostEqual(row0["count_column"], 0, delta=1e-3)
        self.assertAlmostEqual(row0["sum"], 0, delta=1e-3)
        # check row[1] = "key1", 2+noise, 3+noise
        row1 = pandas_df.loc[1]
        self.assertEqual(row1["group_key"], "key1")
        self.assertAlmostEqual(row1["count_column"], 2, delta=1e-3)
        self.assertAlmostEqual(row1["sum"], 3, delta=1e-3)
        self.assertAlmostEqual(row1["mean"], 1.5, delta=1e-3)

    def test_run_query_multiple_partition_keys_e2e_run(self):
        # Arrange
        spark = self._get_spark_session()
        pandas_df = get_pandas_df()
        pandas_df["second_group_by_key"] = ["a", "b", "c"]
        df = spark.createDataFrame(pandas_df)
        columns = dataframes.Columns("privacy_key",
                                     ["group_key", "second_group_by_key"],
                                     "value")
        metrics = {
            pipeline_dp.Metrics.COUNT: None,  # it returns default name "count"
            pipeline_dp.Metrics.SUM: "sum_column"
        }
        bounds = dataframes.ContributionBounds(
            max_partitions_contributed=2,
            max_contributions_per_partition=2,
            min_value=-5,
            max_value=5)
        public_keys = [("key1", "a"), ("key0", "b")]
        query = dataframes.Query(df, columns, metrics, bounds, public_keys)

        # Act
        budget = dataframes.Budget(1e6, 1e-1)  # large budget to get small noise
        result_df = query.run_query(budget)

        # Assert
        pandas_df = result_df.toPandas()
        pandas_df = pandas_df.sort_values(by=['group_key']).reset_index(
            drop=True)
        self.assertLen(pandas_df, 2)
        # check row[0] = "key0", "b", 0+noise, 0+noise
        row0 = pandas_df.loc[0]
        self.assertEqual(row0["group_key"], "key0")
        self.assertEqual(row0["second_group_by_key"], "b")
        self.assertAlmostEqual(row0["count"], 0, delta=1e-3)
        self.assertAlmostEqual(row0["sum_column"], 0, delta=1e-3)
        # check row[1] = "key1", "a", 1+noise, 3+noise
        row1 = pandas_df.loc[1]
        self.assertEqual(row1["group_key"], "key1")
        self.assertEqual(row1["second_group_by_key"], "a")
        self.assertAlmostEqual(row1["count"], 1, delta=1e-3)
        self.assertAlmostEqual(row1["sum_column"], 5, delta=1e-3)

    def test_run_query_e2e_run_empty_result(self):
        # Arrange
        spark = self._get_spark_session()
        df = spark.createDataFrame(get_pandas_df())
        columns = dataframes.Columns("privacy_key", "group_key", "value")
        metrics = {pipeline_dp.Metrics.COUNT: "count_column"}
        bounds = dataframes.ContributionBounds(
            max_partitions_contributed=2,
            max_contributions_per_partition=2,
            min_value=-5,
            max_value=5)
        query = dataframes.Query(df,
                                 columns,
                                 metrics,
                                 bounds,
                                 public_partitions=None)

        # Act
        budget = dataframes.Budget(1, 1e-10)
        result_df = query.run_query(budget)

        # Assert
        # The small input dataset and private partition selection. It almost
        # sure leads to the empty result.
        self.assertTrue(result_df.toPandas().empty)


if __name__ == '__main__':
    absltest.main()
