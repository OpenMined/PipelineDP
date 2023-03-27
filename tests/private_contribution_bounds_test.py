import unittest

import pipeline_dp
from pipeline_dp import NaiveBudgetAccountant, PipelineBackend, \
    CalculatePrivateContributionBoundsParams, PrivateContributionBounds
from pipeline_dp.report_generator import ReportGenerator


class MyTestCase(unittest.TestCase):

    def _create_dp_engine_default(self,
                                  accountant: NaiveBudgetAccountant = None,
                                  backend: PipelineBackend = None,
                                  return_accountant: bool = False):
        if not accountant:
            accountant = NaiveBudgetAccountant(total_epsilon=1,
                                               total_delta=1e-10)
        if not backend:
            backend = pipeline_dp.LocalBackend()
        dp_engine = pipeline_dp.DPEngine(accountant, backend)
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            metrics=[],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)
        dp_engine._report_generators.append(
            ReportGenerator(aggregator_params, "test_method"))
        dp_engine._add_report_stage("DP Engine Test")
        if return_accountant:
            return dp_engine, accountant
        return dp_engine

    def test_calculate_private_contribution_bounds_basic(self):
        # Arrange
        engine, accountant = self._create_dp_engine_default(
            return_accountant=True)
        params = CalculatePrivateContributionBoundsParams(
            aggregation_eps=0.9,
            aggregation_delta=1e-10,
            calculation_eps=0.1,
            aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed_upper_bound=2)
        # user 1 contributes 3 times to partition 0 and 2 times to partitions 1.
        # user 2 contributes 2 times to partition 0 and 3 times to partitions 1.
        data = [("pk0", 1), ("pk0", 1), ("pk0", 1), ("pk0", 2), ("pk1", 2),
                ("pk1", 1), ("pk1", 1), ("pk1", 2), ("pk0", 2), ("pk1", 2)]
        data_extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: x[0],
            privacy_id_extractor=lambda x: x[1],
            value_extractor=lambda _: None,
        )

        # Act
        result = engine.calculate_private_contribution_bounds(
            col=data,
            params=params,
            data_extractors=data_extractors,
            partitions=["pk0", "pk1"])
        result = list(result)[0]

        # Assert
        self.assertEqual(
            result, PrivateContributionBounds(max_partitions_contributed=2))


if __name__ == '__main__':
    unittest.main()
