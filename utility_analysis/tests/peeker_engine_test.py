"""Tests for peeker_dp_engine."""

from absl.testing import absltest
from absl.testing import parameterized
import pipeline_dp

from utility_analysis import peeker_engine


class PeekerDpEngineTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self._pipeline_backend = pipeline_dp.LocalBackend()

    @parameterized.named_parameters(
        dict(testcase_name='low_epsilon',
             sketches=[(f'pk{i%50}', i, 1) for i in range(100)],
             epsilon=1,
             delta=1e-10,
             size_lower_bound=0,
             size_upper_bound=3),
        dict(testcase_name='high_epsilon',
             sketches=[(f'pk{i%10}', 5, 1) for i in range(100)],
             epsilon=10000,
             delta=0.1,
             size_lower_bound=9,
             size_upper_bound=10),
    )
    def test_aggregate_sketches_sum(self, sketches, epsilon, delta,
                                    size_lower_bound, size_upper_bound):
        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            metrics=[pipeline_dp.Metrics.SUM],
            max_partitions_contributed=1,
            max_contributions_per_partition=10,
            min_value=0,
            max_value=10)
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(
            total_epsilon=epsilon, total_delta=delta)
        dp_engine = peeker_engine.PeekerEngine(budget_accountant,
                                               self._pipeline_backend)
        dp_results = dp_engine.aggregate_sketches(sketches, params)
        budget_accountant.compute_budgets()
        dp_results = list(dp_results)
        self.assertLessEqual(len(dp_results), size_upper_bound)
        self.assertGreaterEqual(len(dp_results), size_lower_bound)


if __name__ == '__main__':
    absltest.main()
