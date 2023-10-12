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
            "partition_key": ["key1", "key2", "key1"],
            "value": [5.0, 2.5, -2]
        })

    def test_query_count(self):
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


if __name__ == '__main__':
    absltest.main()
