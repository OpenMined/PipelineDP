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

import pipeline_dp
from pipeline_dp import budget_accounting
from analysis import utility_analysis_engine
from analysis import metrics
import analysis


class UtilityAnalysisEngineTest(parameterized.TestCase):

    def _get_default_extractors(self) -> pipeline_dp.DataExtractors:
        return pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    def _get_default_pre_aggregated_extractors(
            self) -> pipeline_dp.PreAggregateExtractors:
        return pipeline_dp.PreAggregateExtractors(
            partition_extractor=lambda x: x[0],
            preaggregate_extractor=lambda x: x[1])

    def _get_default_aggregate_params(self) -> pipeline_dp.AggregateParams:
        return pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

    def test_invalid_utility_analysis_params_throws_exception(self):
        # Arrange.
        default_extractors = self._get_default_extractors()
        default_params = self._get_default_aggregate_params()
        params_with_custom_combiners = copy.copy(default_params)
        params_with_custom_combiners.custom_combiners = sum
        params_with_unsupported_metric = copy.copy(default_params)
        params_with_unsupported_metric.metrics = [pipeline_dp.Metrics.MEAN]
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
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=1e-10)
            options = analysis.UtilityAnalysisOptions(
                epsilon=1,
                delta=0,
                aggregate_params=test_case["params"],
                pre_aggregated_data=test_case["pre_aggregated"])
            engine = utility_analysis_engine.UtilityAnalysisEngine(
                budget_accountant=budget_accountant,
                backend=pipeline_dp.LocalBackend())
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

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)

        public_partitions = ["pk0", "pk1", "empty_public_partition"]

        # Input collection has 100 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        input = list(range(100))

        if not pre_aggregated:
            data_extractors = pipeline_dp.DataExtractors(
                privacy_id_extractor=lambda x: x,
                partition_extractor=lambda x: f"pk{x}",
                value_extractor=lambda x: 0)
        else:
            data_extractors = pipeline_dp.PreAggregateExtractors(
                partition_extractor=lambda x: f"pk{x}",
                preaggregate_extractor=lambda x: (1, 0, 1))

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

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
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=2)

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=2,
                                                              total_delta=1e-10)
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: f"pk{x[1]}",
            value_extractor=lambda x: 0)

        # Input collection has 10 privacy ids where each privacy id
        # contributes to the same 10 partitions, three times in each partition.
        input = [(i, j) for i in range(10) for j in range(10)] * 3
        if pre_aggregated:
            input = analysis.pre_aggregation.preaggregate(
                input, pipeline_dp.LocalBackend(), data_extractors)
            data_extractors = self._get_default_pre_aggregated_extractors()

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

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
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        multi_param = analysis.MultiParameterConfiguration(
            max_partitions_contributed=[1, 2],
            max_contributions_per_partition=[1, 2])

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = pipeline_dp.DataExtractors(
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
            metrics.RawStatistics(privacy_id_count=2, count=1),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=5.87109375,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=16.60596081442783,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
        ]
        expected_pk1 = [
            metrics.RawStatistics(privacy_id_count=2, count=2),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=2.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=-1.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=5.87109375,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=2.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=16.60596081442783,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
        ]

        self.assertSequenceEqual(expected_pk0, output[0][1])
        self.assertSequenceEqual(expected_pk1, output[1][1])

    def test_multi_parameters_check_different_budget_accounts(self):
        # Arrange
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        multi_param = analysis.MultiParameterConfiguration(
            max_partitions_contributed=[1, 2],
            max_contributions_per_partition=[1, 2],
            noise_kind=[
                pipeline_dp.NoiseKind.LAPLACE, pipeline_dp.NoiseKind.GAUSSIAN
            ])

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = pipeline_dp.DataExtractors(
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
        self.assertLen(output, 2)
        # Each partition has 2 metrics (for both parameter set).
        [self.assertLen(partition_metrics, 2) for partition_metrics in output]

        expected_pk0 = [
            metrics.RawStatistics(privacy_id_count=2, count=1),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=5.87109375,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=1.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=16.60596081442783,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
        ]
        expected_pk1 = [
            metrics.RawStatistics(privacy_id_count=2, count=2),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=2.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=-1.0,
                               expected_l0_bounding_error=-0.5,
                               std_l0_bounding_error=0.5,
                               std_noise=5.87109375,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                               sum=2.0,
                               clipping_to_min_error=0.0,
                               clipping_to_max_error=0.0,
                               expected_l0_bounding_error=0,
                               std_l0_bounding_error=0.0,
                               std_noise=16.60596081442783,
                               noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
        ]

        self.assertSequenceEqual(expected_pk0, output[0][1])
        self.assertSequenceEqual(expected_pk1, output[1][1])

    @patch('pipeline_dp.sampling_utils.ValueSampler.__init__')
    def test_partition_sampling(self, mock_sampler_init):
        # Arrange
        mock_sampler_init.return_value = None
        aggregator_params = self._get_default_aggregate_params()

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)

        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: None)

        engine = utility_analysis_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        options = analysis.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregator_params,
            partitions_sampling_prob=0.25)
        engine.analyze(col=[1, 2, 3],
                       options=options,
                       data_extractors=data_extractors)
        mock_sampler_init.assert_called_once_with(0.25)


if __name__ == '__main__':
    absltest.main()
