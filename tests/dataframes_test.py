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

import pandas as pd
import pipeline_dp
from pipeline_dp import dataframes


class QueryBuilderTest(parameterized.TestCase):

    def _get_pandas_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "privacy_key": [0, 1, 1],
            "group_key": ["key1", "key2", "key1"],
            "value": [5.0, 2.5, -2]
        })

    def test_incorrect_privacy_unit_column(self):
        df = self._get_pandas_df()
        with self.assertRaisesRegex(ValueError,
                                    "Column not_present_column is not present"):
            dataframes.QueryBuilder(df, "not_present_column")

    def test_global_aggregations_are_not_supported(self):
        df = self._get_pandas_df()
        builder = dataframes.QueryBuilder(df, "privacy_key")
        with self.assertRaisesRegex(ValueError,
                                    "Global aggregations are not supported"):
            builder.count()

    def test_group_by_not_existing_column(self):
        df = self._get_pandas_df()
        builder = dataframes.QueryBuilder(df, "privacy_key")
        with self.assertRaisesRegex(
                ValueError,
                "Column not_present_column is not present in DataFrame"):
            builder.groupby("not_present_column",
                            max_groups_contributed=1,
                            max_contributions_per_group=1)

    def test_2_group_by(self):
        df = self._get_pandas_df()
        builder = dataframes.QueryBuilder(df, "privacy_key")
        builder.groupby("group_key",
                        max_groups_contributed=1,
                        max_contributions_per_group=1)
        with self.assertRaisesRegex(ValueError,
                                    "groupby can be called only once"):
            builder.groupby("group_key",
                            max_groups_contributed=1,
                            max_contributions_per_group=1)

    def test_count_query(self):
        df = self._get_pandas_df()
        query = dataframes.QueryBuilder(df, "privacy_key").groupby(
            "partition_key",
            max_groups_contributed=5,
            max_contributions_per_group=10).count().build_query()
        self.assertTrue(query._df.equals(df))
        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key", "partition_key", None))
        self.assertEqual(query._metrics, [pipeline_dp.Metrics.COUNT])
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=5,
                                          max_contributions_per_partition=10))

    def test_sum_query(self):
        df = self._get_pandas_df()
        query = dataframes.QueryBuilder(df, "privacy_key").groupby(
            "partition_key",
            max_groups_contributed=8,
            max_contributions_per_group=11).sum("value",
                                                min_value=1,
                                                max_value=2.5).build_query()
        self.assertTrue(query._df.equals(df))
        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key", "partition_key", "value"))
        self.assertEqual(query._metrics, [pipeline_dp.Metrics.SUM])
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=8,
                                          max_contributions_per_partition=11,
                                          min_value=1,
                                          max_value=2.5))

    def test_count_and_sum_query(self):
        query = dataframes.QueryBuilder(
            self._get_pandas_df(),
            "privacy_key").groupby("partition_key",
                                   max_groups_contributed=8,
                                   max_contributions_per_group=11).count().sum(
                                       "value", min_value=1,
                                       max_value=2.5).build_query()
        self.assertEqual(
            query._columns,
            dataframes.Columns("privacy_key", "partition_key", "value"))
        self.assertEqual(query._metrics,
                         [pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM])
        self.assertEqual(
            query._contribution_bounds,
            dataframes.ContributionBounds(max_partitions_contributed=8,
                                          max_contributions_per_partition=11,
                                          min_value=1,
                                          max_value=2.5))


if __name__ == '__main__':
    absltest.main()
