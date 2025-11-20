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
"""Tests for the advanced query builder API."""

import unittest
from unittest.mock import patch
from pipeline_dp import advanced_query_builder as aqb
from pipeline_dp import aggregate_params as ap
from pipeline_dp import budget_accounting as ba
from pipeline_dp import pipeline_backend


class AdvancedQueryBuilderTest(unittest.TestCase):

    @patch('pipeline_dp.advanced_query_builder.DPEngine')
    def test_local_backend_composite_key(self, mock_dp_engine):
        # Arrange
        data = [
            {
                'privacy_key': 'pk1',
                'partition_key1': 'pk2',
                'partition_key2': 'pk4',
                'value1': 1
            },
            {
                'privacy_key': 'pk1',
                'partition_key1': 'pk2',
                'partition_key2': 'pk4',
                'value1': 2
            },
            {
                'privacy_key': 'pk1',
                'partition_key1': 'pk3',
                'partition_key2': 'pk5',
                'value1': 3
            },
        ]

        # Act
        query = aqb.QueryBuilder() \
            .from_(data) \
            .group_by(
            aqb.ColumnNames(("partition_key1", "partition_key2")),
            aqb.OptimalGroupSelectionGroupBySpec(
                privacy_unit=aqb.ColumnNames("privacy_key"),
                budget=aqb.EpsDeltaBudget(1.0, 1e-10),
                contribution_bounding_level=aqb.ContributionBoundingLevel(
                    max_partitions_contributed=1,
                    max_contributions_per_partition=1
                ),
                pre_threshold=1
            )
        ) \
            .count(
            output_column_name="count",
            spec=aqb.LaplaceCountSpec(budget=aqb.EpsBudget(1.0))
        ) \
            .build()

        query.run()

        # Assert
        mock_dp_engine.assert_called_once()
        _, kwargs = mock_dp_engine.call_args
        self.assertIsInstance(kwargs['budget_accountant'],
                              ba.NaiveBudgetAccountant)
        self.assertIsInstance(kwargs['backend'], pipeline_backend.LocalBackend)

        mock_aggregate = mock_dp_engine.return_value.aggregate
        mock_aggregate.assert_called_once()
        _, kwargs = mock_aggregate.call_args
        self.assertEqual(kwargs['params'].metrics, [ap.Metrics.COUNT])


if __name__ == '__main__':
    unittest.main()
