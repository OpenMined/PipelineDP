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
from utility_analysis_new import dp_engine
from utility_analysis_new import metrics
import utility_analysis_new


class MultiParameterConfiguration(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="All MultiParameterConfiguration fields unset",
             error_msg="MultiParameterConfiguration must have at least 1 "
             "non-empty attribute.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=None,
             max_sum_per_partition=None),
        dict(testcase_name="Attributes different size 1",
             error_msg="All set attributes in MultiParameterConfiguration must "
             "have the same length.",
             max_partitions_contributed=[1],
             max_contributions_per_partition=[1, 2],
             min_sum_per_partition=None,
             max_sum_per_partition=None),
        dict(testcase_name="Attributes different size 2",
             error_msg="All set attributes in MultiParameterConfiguration must "
             "have the same length.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=[1, 1, 1],
             max_sum_per_partition=[2]),
        dict(testcase_name="One of min_sum_per_partition, "
             "max_sum_per_partition is None",
             error_msg="MultiParameterConfiguration: min_sum_per_partition and "
             "max_sum_per_partition must be both set or both None.",
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             min_sum_per_partition=[1, 1, 1],
             max_sum_per_partition=None),
    )
    def test_validation(self, error_msg, max_partitions_contributed,
                        max_contributions_per_partition, min_sum_per_partition,
                        max_sum_per_partition):
        with self.assertRaisesRegex(ValueError, error_msg):
            utility_analysis_new.MultiParameterConfiguration(
                max_partitions_contributed, max_contributions_per_partition,
                min_sum_per_partition, max_sum_per_partition)

    def test_get_aggregate_params(self):
        params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        max_partitions_contributed = [10, 12, 15]
        multi_params = utility_analysis_new.MultiParameterConfiguration(
            max_partitions_contributed=max_partitions_contributed)
        self.assertTrue(3, multi_params.size)

        for i in range(multi_params.size):
            ith_params = multi_params.get_aggregate_params(params, i)
            params.max_partitions_contributed = max_partitions_contributed[i]
            self.assertEqual(params, ith_params)


class DpEngine(parameterized.TestCase):

    def _get_default_extractors(self) -> pipeline_dp.DataExtractors:
        return pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    def _get_default_aggregate_params(self) -> pipeline_dp.AggregateParams:
        return pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

    def test_utility_analysis_params(self):
        default_extractors = self._get_default_extractors()
        default_params = self._get_default_aggregate_params()
        params_with_custom_combiners = copy.copy(default_params)
        params_with_custom_combiners.custom_combiners = sum
        params_with_unsupported_metric = copy.copy(default_params)
        params_with_unsupported_metric.metrics = [pipeline_dp.Metrics.MEAN]
        params_with_contribution_bounds_already_enforced = default_params
        params_with_contribution_bounds_already_enforced.contribution_bounds_already_enforced = True

        test_cases = [
            {
                "desc": "custom combiners",
                "params": params_with_custom_combiners,
                "data_extractor": default_extractors,
                "public_partitions": [1]
            },
            {
                "desc": "unsupported metric in metrics",
                "params": params_with_unsupported_metric,
                "data_extractor": default_extractors,
                "public_partitions": [1]
            },
            {
                "desc": "contribution bounds are already enforced",
                "params": params_with_contribution_bounds_already_enforced,
                "data_extractor": default_extractors,
                "public_partitions": [1]
            },
        ]

        for test_case in test_cases:

            with self.assertRaisesRegex(Exception,
                                        expected_regex=test_case["desc"]):
                budget_accountant = budget_accounting.NaiveBudgetAccountant(
                    total_epsilon=1, total_delta=1e-10)
                options = utility_analysis_new.UtilityAnalysisOptions(
                    epsilon=1, delta=0, aggregate_params=test_case["params"])
                engine = dp_engine.UtilityAnalysisEngine(
                    budget_accountant=budget_accountant,
                    backend=pipeline_dp.LocalBackend())
                col = [0, 1, 2]

                engine.analyze(col,
                               options,
                               test_case["data_extractor"],
                               public_partitions=test_case["public_partitions"])

    def test_aggregate_public_partition_e2e(self):
        # Arrange
        aggregator_params = self._get_default_aggregate_params()

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)

        public_partitions = ["pk0", "pk1", "pk101"]

        # Input collection has 100 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        col = list(range(100))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: None)

        engine = dp_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        options = utility_analysis_new.UtilityAnalysisOptions(
            epsilon=1, delta=0, aggregate_params=aggregator_params)
        col = engine.analyze(col=col,
                             options=options,
                             data_extractors=data_extractor,
                             public_partitions=public_partitions)
        budget_accountant.compute_budgets()

        col = list(col)

        # Assert public partitions are applied.
        self.assertLen(col, 3)
        self.assertTrue(any(v[0] == 'pk101' for v in col))

    def test_aggregate_error_metrics(self):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=2)

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=2,
                                                              total_delta=1e-10)

        # Input collection has 10 privacy ids where each privacy id
        # contributes to the same 10 partitions, three times in each partition.
        col = [(i, j) for i in range(10) for j in range(10)] * 3
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: f"pk{x[1]}",
            value_extractor=lambda x: None)

        engine = dp_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        options = utility_analysis_new.UtilityAnalysisOptions(
            epsilon=1, delta=0, aggregate_params=aggregator_params)
        col = engine.analyze(col=col,
                             options=options,
                             data_extractors=data_extractors)
        budget_accountant.compute_budgets()

        col = list(col)

        # Assert
        self.assertLen(col, 10)
        # Assert count metrics are correct.
        [self.assertTrue(v[1][1].per_partition_error == -10) for v in col]
        [
            self.assertAlmostEqual(v[1][1].expected_cross_partition_error,
                                   -18.0,
                                   delta=1e-5) for v in col
        ]
        [
            self.assertAlmostEqual(v[1][1].std_cross_partition_error,
                                   1.89736,
                                   delta=1e-5) for v in col
        ]
        [
            self.assertAlmostEqual(v[1][1].std_noise, 11.95312, delta=1e-5)
            for v in col
        ]

    def test_multi_parameters(self):
        # Arrange
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        multi_param = utility_analysis_new.MultiParameterConfiguration(
            max_partitions_contributed=[1, 2],
            max_contributions_per_partition=[1, 2])

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)

        engine = dp_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: None)

        public_partitions = ["pk0", "pk1"]

        options = utility_analysis_new.UtilityAnalysisOptions(
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
            metrics.CountMetrics(count=1,
                                 per_partition_error=0,
                                 expected_cross_partition_error=-0.5,
                                 std_cross_partition_error=0.5,
                                 std_noise=11.6640625,
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            metrics.CountMetrics(count=1,
                                 per_partition_error=0,
                                 expected_cross_partition_error=0,
                                 std_cross_partition_error=0.0,
                                 std_noise=32.99095075973487,
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
        ]
        expected_pk1 = [
            metrics.CountMetrics(count=2,
                                 per_partition_error=-1,
                                 expected_cross_partition_error=-0.5,
                                 std_cross_partition_error=0.5,
                                 std_noise=11.6640625,
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN),
            metrics.CountMetrics(count=2,
                                 per_partition_error=0,
                                 expected_cross_partition_error=0,
                                 std_cross_partition_error=0.0,
                                 std_noise=32.99095075973487,
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

        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: None)

        engine = dp_engine.UtilityAnalysisEngine(
            budget_accountant=budget_accountant,
            backend=pipeline_dp.LocalBackend())

        options = utility_analysis_new.UtilityAnalysisOptions(
            epsilon=1,
            delta=0,
            aggregate_params=aggregator_params,
            partitions_sampling_prob=0.25)
        engine.analyze(col=[1, 2, 3],
                       options=options,
                       data_extractors=data_extractor)
        mock_sampler_init.assert_called_once_with(0.25)


if __name__ == '__main__':
    absltest.main()
