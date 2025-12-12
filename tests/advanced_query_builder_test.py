import unittest
from pipeline_dp import advanced_query_builder as aqb


class AdvancedQueryBuilderTest(unittest.TestCase):

    def test_e2e_local_backend(self):
        # Input data: list of dictionaries
        # m1: u1(5), u2(4), u3(3)
        # m2: u1(2), u4(1)
        data = [
            {
                'movie_id': 'm1',
                'user_id': 'u1',
                'rating': 5
            },
            {
                'movie_id': 'm1',
                'user_id': 'u2',
                'rating': 4
            },
            {
                'movie_id': 'm1',
                'user_id': 'u3',
                'rating': 3
            },
            {
                'movie_id': 'm2',
                'user_id': 'u1',
                'rating': 2
            },
            {
                'movie_id': 'm2',
                'user_id': 'u4',
                'rating': 1
            },
        ]

        # Define Query
        query = aqb.QueryBuilder().from_(data).group_by(
            aqb.ColumnNames("movie_id"),
            aqb.OptimalGroupSelectionGroupBySpec.Builder().setPrivacyUnit(
                aqb.ColumnNames("user_id")).setContributionBoundingLevel(
                    aqb.PartitionLevel(
                        max_partitions_contributed=3,
                        max_contributions_per_partition=1)).setBudget(
                            eps=1.0).setPublicGroups(["m1", "m2"]).build()
        ).count(
            "rating_count",
            aqb.LaplaceCountSpec.Builder().setPrivacyUnit(
                aqb.ColumnNames("user_id")).setBudget(
                    eps=1.0).setContributionBoundingLevel(
                        aqb.PartitionLevel(
                            max_partitions_contributed=3,
                            max_contributions_per_partition=1)).build()
        ).count_distinct(
            aqb.ColumnNames("user_id"),
            "user_count",
            aqb.LaplaceCountDistinctSpec.Builder().setPrivacyUnit(
                aqb.ColumnNames("user_id")).setBudget(
                    eps=1.0).setContributionBoundingLevel(
                        aqb.PartitionLevel(
                            max_partitions_contributed=3,
                            max_contributions_per_partition=1)).build()
        ).sum(
            "rating", "rating_sum",
            aqb.LaplaceSumSpec.Builder().setPrivacyUnit(
                aqb.ColumnNames("user_id")).setBudget(
                    eps=1.0).setContributionBoundingLevel(
                        aqb.PartitionLevel(max_partitions_contributed=3,
                                           max_contributions_per_partition=1)).
            setTotalValueBounds(
                minTotalValue=0, maxTotalValue=100).build()).mean(
                    "rating", "rating_avg",
                    aqb.LaplaceMeanSpec.Builder().setPrivacyUnit(
                        aqb.ColumnNames("user_id")).setBudget(
                            eps=1.0).setContributionBoundingLevel(
                                aqb.PartitionLevel(
                                    max_partitions_contributed=3,
                                    max_contributions_per_partition=1)).
                    setValueBounds(minValue=1, maxValue=5).build()).variance(
                        "rating", "rating_var",
                        aqb.LaplaceVarianceSpec.Builder().setPrivacyUnit(
                            aqb.ColumnNames("user_id")).setBudget(
                                eps=1.0).setContributionBoundingLevel(
                                    aqb.PartitionLevel(
                                        max_partitions_contributed=3,
                                        max_contributions_per_partition=1)).
                        setValueBounds(minValue=1, maxValue=5).build()
                    ).quantiles(
                        "rating",
                        [0.5, 0.9],
                        "rating_quantiles",
                        aqb.TreeQuantilesSpec.Builder().setPrivacyUnit(
                            aqb.ColumnNames("user_id")).setBudget(
                                eps=1.0,
                                delta=1e-6).setContributionBoundingLevel(
                                    aqb.PartitionLevel(
                                        max_partitions_contributed=3,
                                        max_contributions_per_partition=1)
                                ).setValueBounds(minValue=1,
                                                 maxValue=5).build()).build()

        # Run Query
        results = query.run()

        # Verification (Approximate since it's DP)
        # We expect 2 rows: one for m1, one for m2
        self.assertEqual(len(results), 2)

        m1_row = next((r for r in results if r['movie_id'] == 'm1'), None)
        m2_row = next((r for r in results if r['movie_id'] == 'm2'), None)

        self.assertIsNotNone(m1_row)
        self.assertIsNotNone(m2_row)

        # Check presence and types of all metrics
        # Count
        self.assertTrue(isinstance(m1_row['rating_count'], (int, float)))

        # Count Distinct
        self.assertTrue(isinstance(m1_row['user_count'], (int, float)))

        # Sum
        self.assertTrue(isinstance(m1_row['rating_sum'], (int, float)))

        # Mean
        self.assertTrue(isinstance(m1_row['rating_avg'], (int, float)))

        # Variance
        self.assertTrue(isinstance(m1_row['rating_var'], (int, float)))

        # Quantiles - expects a dictionary with percentile_50 and percentile_90
        self.assertTrue(isinstance(m1_row['rating_quantiles'], dict))
        self.assertIn('percentile_50', m1_row['rating_quantiles'])
        self.assertIn('percentile_90', m1_row['rating_quantiles'])
        self.assertTrue(
            isinstance(m1_row['rating_quantiles']['percentile_50'],
                       (int, float)))

        print(f"M1 Row: {m1_row}")


if __name__ == '__main__':
    unittest.main()
