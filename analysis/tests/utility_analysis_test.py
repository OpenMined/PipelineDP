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
"""Utility Analysis Public API Test"""
from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
import analysis
from analysis import metrics
from analysis.tests import common


class UtilityAnalysis(parameterized.TestCase):

    def _get_default_extractors(self) -> pipeline_dp.DataExtractors:
        return pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    @parameterized.parameters(False, True)
    def test_wo_public_partitions(self, pre_aggregated: bool):
        # Arrange
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[
                pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                pipeline_dp.Metrics.SUM
            ],
            max_partitions_contributed=1,
            min_sum_per_partition=0,
            max_sum_per_partition=1,
            max_contributions_per_partition=2)

        # Input collection has 10 privacy ids where each privacy id
        # contributes to the same 10 partitions, three times in each partition.
        if not pre_aggregated:
            col = [(i, j) for i in range(10) for j in range(10)] * 3
        else:
            # This is pre-aggregated dataset, namely each element has a format
            # (partition_key, (count, sum, num_partition_contributed).
            # And each element is in one-to-one correspondence to pairs
            # (privacy_id, partition_key) from the dataset.
            col = [(i, (3, 1, 10)) for i in range(10)] * 10

        if not pre_aggregated:
            data_extractors = pipeline_dp.DataExtractors(
                privacy_id_extractor=lambda x: x[0],
                partition_extractor=lambda x: f"pk{x[1]}",
                value_extractor=lambda x: 1)
        else:
            data_extractors = analysis.PreAggregateExtractors(
                partition_extractor=lambda x: f"pk{x[0]}",
                preaggregate_extractor=lambda x: x[1])

        col = analysis.perform_utility_analysis(
            col=col,
            backend=pipeline_dp.LocalBackend(),
            options=analysis.UtilityAnalysisOptions(
                epsilon=3,
                delta=0.9,
                aggregate_params=aggregate_params,
                pre_aggregated_data=pre_aggregated),
            data_extractors=data_extractors)

        col = list(col)

        # Assert
        # Assert a singleton is returned
        self.assertLen(col, 1)
        self.assertLen(col[0], 1)
        output = col[0][0]
        # Assert partition selection metrics are reasonable.
        # partition_kept_probability = 0.4311114 for each partition
        expected_partition_selection_metrics = metrics.PartitionSelectionMetrics(
            num_partitions=10,
            dropped_partitions_expected=7.08783,
            dropped_partitions_variance=2.06410)
        common.assert_dataclasses_are_equal(
            self, expected_partition_selection_metrics,
            output.partition_selection_metrics)
        # Assert count metrics are reasonable.
        expected_count_metrics = metrics.AggregateErrorMetrics(
            metric_type=metrics.AggregateMetricType.COUNT,
            ratio_data_dropped_l0=0.6,
            ratio_data_dropped_linf=0.33333,
            ratio_data_dropped_partition_selection=0.04725,
            error_l0_expected=-18.0,
            error_linf_expected=-10.0,
            error_linf_min_expected=0.0,
            error_linf_max_expected=-10.0,
            error_expected=-28.0,
            error_l0_variance=3.6,
            error_variance=6.78678,
            error_quantiles=[-24.66137],
            rel_error_l0_expected=-0.6,
            rel_error_linf_expected=-0.33333,
            rel_error_linf_min_expected=0.0,
            rel_error_linf_max_expected=-0.33333,
            rel_error_expected=-0.93333,
            rel_error_l0_variance=0.004,
            rel_error_variance=0.00754,
            rel_error_quantiles=[-0.82204],
            error_expected_w_dropped_partitions=-29.41756,
            rel_error_expected_w_dropped_partitions=-0.98058,
            noise_std=1.78515,
        )
        common.assert_dataclasses_are_equal(self, expected_count_metrics,
                                            output.count_metrics)

    @parameterized.named_parameters(
        dict(
            testcase_name="Gaussian noise",
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            expected=metrics.AggregateErrorMetrics(
                metric_type=metrics.AggregateMetricType.COUNT,
                ratio_data_dropped_l0=0,
                ratio_data_dropped_linf=0,
                ratio_data_dropped_partition_selection=0,
                error_l0_expected=0,
                error_linf_expected=0,
                error_linf_min_expected=0.0,
                error_linf_max_expected=0.0,
                error_expected=0,
                error_l0_variance=0.0,
                error_variance=35.71929,
                error_quantiles=[7.65927, 0.0],
                rel_error_l0_expected=0,
                rel_error_linf_expected=0,
                rel_error_linf_min_expected=0.0,
                rel_error_linf_max_expected=0.0,
                rel_error_expected=0,
                rel_error_l0_variance=0.0,
                rel_error_variance=23.81286,
                rel_error_quantiles=[5.10618, 0.0],
                error_expected_w_dropped_partitions=0.0,
                rel_error_expected_w_dropped_partitions=0.0,
                noise_std=5.97656,
            ),
        ),
        dict(
            testcase_name="Laplace noise",
            noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            expected=metrics.AggregateErrorMetrics(
                metric_type=metrics.AggregateMetricType.COUNT,
                ratio_data_dropped_l0=0.0,
                ratio_data_dropped_linf=0.0,
                ratio_data_dropped_partition_selection=0.0,
                error_l0_expected=0.0,
                error_linf_expected=0.0,
                error_linf_min_expected=0.0,
                error_linf_max_expected=0.0,
                error_expected=0.0,
                error_l0_variance=0.0,
                error_variance=2.0,
                error_quantiles=[1.58, 0.0],
                rel_error_l0_expected=0,
                rel_error_linf_expected=0,
                rel_error_linf_min_expected=0.0,
                rel_error_linf_max_expected=0.0,
                rel_error_expected=0.0,
                rel_error_l0_variance=0.0,
                rel_error_variance=1.3333333333333334,
                rel_error_quantiles=[1.1, 0.0],
                error_expected_w_dropped_partitions=0.0,
                rel_error_expected_w_dropped_partitions=0.0,
                noise_std=1.41421,
            ),
        ),
    )
    def test_w_public_partitions(self, noise_kind, expected):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=noise_kind,
            metrics=[
                pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT
            ],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        public_partitions = ["pk0", "pk1", "pk101"]

        # Input collection has 100 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        col = list(range(100))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: 0)

        col = analysis.perform_utility_analysis(
            col=col,
            backend=pipeline_dp.LocalBackend(),
            options=analysis.UtilityAnalysisOptions(
                epsilon=2, delta=1e-10, aggregate_params=aggregator_params),
            data_extractors=data_extractor,
            public_partitions=public_partitions)

        col = list(col)

        # Assert
        # Assert a singleton is returned
        self.assertLen(col, 1)
        # Assert there is only a single AggregateMetrics & no metrics for
        # partition selection are returned.
        self.assertLen(col[0], 1)

        # Assert count & privacy id count metrics are reasonable.
        # Using large delta because error quantiles for Laplace are not very
        # accurate.
        common.assert_dataclasses_are_equal(self, expected,
                                            col[0][0].count_metrics, 0.5)
        expected.metric_type = metrics.AggregateMetricType.PRIVACY_ID_COUNT
        common.assert_dataclasses_are_equal(self, expected,
                                            col[0][0].privacy_id_count_metrics,
                                            0.5)

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

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: 0)

        public_partitions = ["pk0", "pk1"]

        output = analysis.perform_utility_analysis(
            col=input,
            backend=pipeline_dp.LocalBackend(),
            options=analysis.UtilityAnalysisOptions(
                epsilon=2,
                delta=1e-10,
                aggregate_params=aggregate_params,
                multi_param_configuration=multi_param),
            data_extractors=data_extractors,
            public_partitions=public_partitions,
        )

        output = list(output)

        # Assert
        # Check that a singleton is returned
        self.assertLen(output, 1)
        # Check that there are 2 AggregateMetrics returned
        metrics = output[0]
        self.assertLen(metrics, 2)
        # Check error_expected is correct
        self.assertAlmostEqual(metrics[0].count_metrics.error_expected, -1)
        self.assertAlmostEqual(metrics[1].count_metrics.error_expected, 0)
        # Check that AggregateParams in output is correct
        for i in range(2):
            self.assertEqual(
                multi_param.max_partitions_contributed[i],
                metrics[i].input_aggregate_params.max_partitions_contributed)
            self.assertEqual(
                multi_param.max_contributions_per_partition[i], metrics[i].
                input_aggregate_params.max_contributions_per_partition)


if __name__ == '__main__':
    absltest.main()
