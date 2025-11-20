# Copyright 2022 OpenMined.
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
"""UtilityAnalysisEngine Test"""
from absl.testing import absltest
from absl.testing import parameterized
from unittest.mock import patch
import copy

from pipeline_dp.aggregate_params import AggregateParams, Metrics, NoiseKind
from pipeline_dp.data_extractors import PreAggregateExtractors, DataExtractors, MultiValueDataExtractors
from pipeline_dp.budget_accounting import NaiveBudgetAccountant
from pipeline_dp.pipeline_backend import LocalBackend
from analysis import utility_analysis_engine
from analysis import metrics
import analysis
import analysis.contribution_bounders as utility_contribution_bounders


class UtilityAnalysisEngineTest(parameterized.TestCase):

    def _get_default_extractors(self) -> DataExtractors:
        return DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    def _get_default_pre_aggregated_extractors(
            self) -> PreAggregateExtractors:
        return PreAggregateExtractors(
            partition_extractor=lambda x: x[0],
            preaggregate_extractor=lambda x: x[1])

    def _get_default_aggregate_params(self) -> AggregateParams:
        return AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

    def _get_default_utility_engine(self):
        return utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=self._get_default_budget_accountant(),
            backend=LocalBackend())

    def _get_default_budget_accountant(
            self) -> NaiveBudgetAccountant:
        return NaiveBudgetAccountant(total_epsilon=1,
                                                 total_delta=1e-10)

    def test_invalid_utility_analysis_params_throws_exception(self):
        # Arrange.
        default_extractors = self._get_default_extractors()
        default_params = self._get_default_aggregate_params()
        params_with_custom_combiners = copy.copy(default_params)
        params_with_custom_combiners.custom_combiners = sum
        params_with_unsupported_metric = copy.copy(default_params)
        params_with_unsupported_metric.metrics = [Metrics.MEAN]
        params_with_contribution_bounds_already_enforced = copy.copy(
            default_params)
        params_with_contribution_bounds_already_enforced.contribution_bounds_already_enforced = True

        test_cases = [{
            "error_message": "custom combiners",
            "params": params_with_custom_combiners,
            "data_extractors": default_extractors,
            "public_partitions": [1],
            "pre_aggregated": False
        }, {
            "error_message": "unsupported metric in metrics",
            "params": params_with_unsupported_metric,
            "data_extractors": default_extractors,
            "public_partitions": [1],
            "pre_aggregated": False
        }, {
            "error_message": "contribution bounds are already enforced",
            "params": params_with_contribution_bounds_already_enforced,
            "data_extractors": default_extractors,
            "public_partitions": [1],
            "pre_aggregated": False
        }, {
            "error_message":
                "PreAggregateExtractors should be specified for pre-aggregated data",
            "params":
                default_params,
            "data_extractors":
                default_extractors,
            "public_partitions": [1],
            "pre_aggregated":
                True
        }]

        for test_case in test_cases:
            budget_accountant = self._get_default_budget_accountant()
            options = analysis.UtilityAnalysisOptions(
                epsilon=1,
                delta=0,
                aggregate_params=test_case["params"],
                pre_aggregated_data=test_case["pre_aggregated"])
            engine = utility_analysis_engine.UtilityAnalysisEngine(
                budget_accountant=budget_accountant,
                backend=LocalBackend())
            # Act and assert.
            with self.assertRaisesRegex(
                    Exception, expected_regex=test_case["error_message"]):
                engine.analyze([0, 1, 2],
                               options,
                               test_case["data_extractors"],
                               public_partitions=test_case["public_partitions"])

    @parameterized.parameters(False, True)
    def test_analyze_applies_public_partitions(self, pre_aggregated: bool):
        # Arrange
        aggregator_params = self._get_default_aggregate_params()

        budget_accountant = self._get_default_budget_accountant()

        public_partitions = ["pk0", "pk1", "empty_public_partition"]

        # Input collection has 100 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        input = list(range(100))

        if not pre_aggregated:
            data_extractors = DataExtractors(
                privacy_id_extractor=lambda x: x,
                partition_extractor=lambda x: f"pk{x}",
                value_extractor=lambda x: 0)
        else:
            data_extractors = PreAggregateExtractors(
                partition_extractor=lambda x: f"pk{x}",
                preaggregate_extractor=lambda x: (1, 0, 1))

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=LocalBackend())

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregator_params,
            pre_aggregated_data=pre_aggregated)
        # Act.
        output = engine.analyze(col=input,
                                options=options,
                                data_extractors=data_extractors,
                                public_partitions=public_partitions)
        budget_accountant.compute_budgets()

        output = list(output)

        # Assert public partitions are applied.
        self.assertLen(output, 3)
        self.assertTrue(any(v[0] == 'empty_public_partition' for v in output))

    @parameterized.parameters(False, True)
    def test_per_partition_error_metrics(self, pre_aggregated: bool):
        # Arrange
        aggregator_params = AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=2)

        budget_accountant = NaiveBudgetAccountant(total_epsilon=2,
                                                              total_delta=1e-10)

        data_extractors = DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: f"pk{x[1]}",
            value_extractor=lambda x: 0)

        # Input collection has 10 privacy ids where each privacy id
        # contributes to the same 10 partitions, three times in each partition.
        input = [(i, j) for i in range(10) for j in range(10)] * 3
        if pre_aggregated:
            input = analysis.pre_aggregation.preaggregate(
                input, LocalBackend(), data_extractors)
            data_extractors = self._get_default_pre_aggregated_extractors()

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=LocalBackend())

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregator_params,
            pre_aggregated_data=pre_aggregated)
        output = engine.analyze(col=input,
                                options=options,
                                data_extractors=data_extractors)
        budget_accountant.compute_budgets()

        output = list(output)

        # Assert
        self.assertLen(output, 10)
        # Assert count metrics are correct.
        [self.assertEqual(v[1][2].clipping_to_max_error, -10) for v in output]
        [
            self.assertAlmostEqual(v[1][2].expected_l0_bounding_error,
                                   -18.0,
                                   delta=1e-5) for v in output
        ]
        [
            self.assertAlmostEqual(v[1][2].std_l0_bounding_error,
                                   1.89736,
                                   delta=1e-5) for v in output
        ]
        [
            self.assertAlmostEqual(v[1][2].std_noise, 11.95312, delta=1e-5)
            for v in output
        ]

    def test_multi_parameters(self):
        # Arrange
        aggregate_params = AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        multi_param = analysis.MultiParameterConfiguration(
            max_partitions_contributed=[1, 2],
            max_contributions_per_partition=[1, 2])

        budget_accountant = self._get_default_budget_accountant()

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=LocalBackend())

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: 0)

        public_partitions = ["pk0", "pk1"]

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregate_params,
            multi_param_configuration=multi_param)
        output = engine.analyze(input,
                                options=options,
                                data_extractors=data_extractors,
                                public_partitions=public_partitions)

        budget_accountant.compute_budgets()

        output = list(output)
        self.assertLen(output, 2)
        # Each partition has 2 metrics (for both parameter set).
        [self.assertLen(partition_metrics, 2) for partition_metrics in output]

        expected_pk0 = [
            metrics.RawStatistics(privacy_id_count=1, count=1),
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=5.87109375,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=16.60596081442783,
                               noise_kind=NoiseKind.GAUSSIAN)
        ]
        expected_pk1 = [
            metrics.RawStatistics(privacy_id_count=1, count=2),
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=2.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=-1.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=5.87109375,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=2.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=16.60596081442783,
                               noise_kind=NoiseKind.GAUSSIAN)
        ]

        self.assertSequenceEqual(expected_pk0, output[0][1])
        self.assertSequenceEqual(expected_pk1, output[1][1])

    def test_multi_parameters_different_noise_kind(self):
        # Arrange
        aggregate_params = AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        multi_param = analysis.MultiParameterConfiguration(
            max_partitions_contributed=[1, 2],
            max_contributions_per_partition=[1, 2],
            noise_kind=[
                NoiseKind.LAPLACE, NoiseKind.GAUSSIAN
            ])

        budget_accountant = self._get_default_budget_accountant()

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=LocalBackend())

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: 0)

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregate_params,
            multi_param_configuration=multi_param)
        output = engine.analyze(input,
                                options=options,
                                data_extractors=data_extractors,
                                public_partitions=None)

        output = list(output)
        self.assertLen(output, 2)  # 2 partitions

        expected_pk0 = [
            metrics.RawStatistics(privacy_id_count=1, count=1),
            5e-11,  # Probability that the partition is kept
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=2.8284271247461903,
                               noise_kind=NoiseKind.LAPLACE),
            2.50000000003125e-11,  # Probability that the partition is kept
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=32.99095075973487,
                               noise_kind=NoiseKind.GAUSSIAN)
        ]

        self.assertSequenceEqual(expected_pk0, output[0][1])

    @patch('pipeline_dp.sampling_utils.ValueSampler.__init__')
    def test_partition_sampling(self, mock_sampler_init):
        # Arrange
        mock_sampler_init.return_value = None
        aggregator_params = self._get_default_aggregate_params()

        budget_accountant = self._get_default_budget_accountant()

        data_extractors = DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: None)

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=LocalBackend())

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregator_params,
            partitions_sampling_prob=0.25)
        engine.analyze(col=[1, 2, 3],
                       options=options,
                       data_extractors=data_extractors)
        mock_sampler_init.assert_called_once_with(0.25)

    def test_utility_analysis_for_2_columns(self):
        # Arrange
        aggregate_params = AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[Metrics.COUNT, Metrics.SUM],
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            max_sum_per_partition=0.5,
            min_sum_per_partition=0)

        multi_param = analysis.MultiParameterConfiguration(
            max_partitions_contributed=[1, 2],
            min_sum_per_partition=[(0, 0), (0, 1)],
            max_sum_per_partition=[(3, 10), (5, 20)],
        )

        budget_accountant = self._get_default_budget_accountant()

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=LocalBackend())

        # Input collection has 2 privacy id, which contributes to 1 partition
        # 2 and 1 times correspondingly.
        input = [(0, "pk", 2, 3), (0, "pk", 0, 0), (1, "pk", 15, 20)]
        data_extractors = MultiValueDataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractors=[lambda x: x[2], lambda x: x[3]])

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregate_params,
            multi_param_configuration=multi_param)
        output = engine.analyze(input,
                                options=options,
                                data_extractors=data_extractors,
                                public_partitions=["pk"])

        budget_accountant.compute_budgets()

        output = list(output)
        self.assertLen(output, 1)
        # Each partition has 2 metrics (for both parameter set).
        [self.assertLen(partition_metrics, 2) for partition_metrics in output]

        expected = [
            metrics.RawStatistics(privacy_id_count=2, count=3),
            metrics.SumMetrics(aggregation=Metrics.SUM,
                               sum=17,
                               clipping_to_min_error=0,
                               clipping_to_max_error=-12,
                               expected_l0_bounding_error=0.0,
                               std_l0_bounding_error=0.0,
                               std_noise=52.359375,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.SUM,
                               sum=23,
                               clipping_to_min_error=0,
                               clipping_to_max_error=-10,
                               expected_l0_bounding_error=0.0,
                               std_l0_bounding_error=0.0,
                               std_noise=174.53125,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=3,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=-1.0,
                               expected_l0_bounding_error=0.0,
                               std_l0_bounding_error=0.0,
                               std_noise=17.453125,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.SUM,
                               sum=17,
                               clipping_to_min_error=0,
                               clipping_to_max_error=-10,
                               expected_l0_bounding_error=0.0,
                               std_l0_bounding_error=0.0,
                               std_noise=123.41223040396463,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.SUM,
                               sum=23,
                               clipping_to_min_error=0,
                               clipping_to_max_error=0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=493.6489216158585,
                               noise_kind=NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=Metrics.COUNT,
                               sum=3,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=-1.0,
                               expected_l0_bounding_error=0.0,
                               std_l0_bounding_error=0.0,
                               std_noise=24.682446080792925,
                               noise_kind=NoiseKind.GAUSSIAN),
        ]

        self.assertSequenceEqual(output[0][1], expected)

    def test_create_contribution_bounder_preaggregated(self):
        engine = self._get_default_utility_engine()
        params = self._get_default_aggregate_params()
        engine._options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=params,
            pre_aggregated_data=True)
        self.assertIsInstance(
            engine._create_contribution_bounder(params, False),
            utility_contribution_bounders.NoOpContributionBounder)

    def test_create_contribution_bounder_l0linf(self):
        engine = self._get_default_utility_engine()
        params = self._get_default_aggregate_params()

        engine._options = analysis.UtilityAnalysisOptions(
            epsilon=1, delta=0, aggregate_params=params)
        self.assertIsInstance(
            engine._create_contribution_bounder(params, False),
            utility_contribution_bounders.L0LinfAnalysisContributionBounder)

    def test_create_contribution_bounder_linf(self):
        engine = self._get_default_utility_engine()
        params = self._get_default_aggregate_params()
        params.perform_cross_partition_contribution_bounding = False

        engine._options = analysis.UtilityAnalysisOptions(
            epsilon=1, delta=0, aggregate_params=params)
        self.assertIsInstance(
            engine._create_contribution_bounder(params, False),
            utility_contribution_bounders.LinfAnalysisContributionBounder)


if __name__ == '__main__':
    absltest.main()
