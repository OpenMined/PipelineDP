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
        self.assertEqual(output.partition_selection_metrics.num_partitions, 10)
        self.assertAlmostEqual(
            output.partition_selection_metrics.dropped_partitions_expected,
            7.08783,
            delta=1e-5)
        self.assertAlmostEqual(
            output.partition_selection_metrics.dropped_partitions_variance,
            2.06410,
            delta=1e-5)
        # Assert count metrics are reasonable.
        self.assertAlmostEqual(output.count_metrics.ratio_data_dropped_l0,
                               0.6,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.ratio_data_dropped_linf,
                               0.33333,
                               delta=1e-5)
        self.assertAlmostEqual(
            output.count_metrics.ratio_data_dropped_partition_selection,
            0.04725,
            delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.error_l0_expected,
                               -18,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.error_linf_expected,
                               -10,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.error_expected,
                               -28,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.error_l0_variance,
                               3.6,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.error_variance,
                               6.78678,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.rel_error_l0_expected,
                               -0.6,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.rel_error_linf_expected,
                               -0.33333,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.rel_error_expected,
                               -0.93333,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.rel_error_l0_variance,
                               0.004,
                               delta=1e-5)
        self.assertAlmostEqual(output.count_metrics.rel_error_variance,
                               0.00754,
                               delta=1e-5)

    @parameterized.named_parameters(
        dict(testcase_name="Gaussian noise",
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             ratio_data_dropped_l0=0,
             ratio_data_dropped_linf=0,
             ratio_data_dropped_partition_selection=0,
             error_l0_expected=0,
             error_linf_expected=0,
             error_expected=0,
             error_l0_variance=0.0,
             error_variance=35.71929,
             rel_error_l0_expected=0,
             rel_error_linf_expected=0,
             rel_error_expected=0,
             rel_error_l0_variance=0.0,
             rel_error_variance=23.81286,
             median_error=0),
        dict(testcase_name="Laplace noise",
             noise_kind=pipeline_dp.NoiseKind.LAPLACE,
             ratio_data_dropped_l0=0,
             ratio_data_dropped_linf=0,
             ratio_data_dropped_partition_selection=0,
             error_l0_expected=0,
             error_linf_expected=0,
             error_expected=0,
             error_l0_variance=0.0,
             error_variance=2.0,
             rel_error_l0_expected=0,
             rel_error_linf_expected=0,
             rel_error_expected=0,
             rel_error_l0_variance=0.0,
             rel_error_variance=1.3333333333333334,
             median_error=0),
    )
    def test_w_public_partitions(
            self, noise_kind, ratio_data_dropped_l0, ratio_data_dropped_linf,
            ratio_data_dropped_partition_selection, error_l0_expected,
            error_linf_expected, error_expected, error_l0_variance,
            error_variance, rel_error_l0_expected, rel_error_linf_expected,
            rel_error_expected, rel_error_l0_variance, rel_error_variance,
            median_error):
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

        # Assert privacy id count metrics are reasonable.
        def check_metric(metrics):
            self.assertEqual(metrics.error_l0_expected, error_l0_expected)
            self.assertEqual(metrics.error_l0_expected, error_linf_expected)
            self.assertEqual(metrics.error_expected, error_expected)
            self.assertAlmostEqual(metrics.error_l0_variance,
                                   error_l0_variance,
                                   delta=1e-5)
            self.assertAlmostEqual(metrics.error_variance,
                                   error_variance,
                                   delta=1e-5)
            self.assertAlmostEqual(metrics.error_quantiles[1],
                                   median_error,
                                   delta=0.1)
            self.assertEqual(metrics.rel_error_l0_expected,
                             rel_error_l0_expected)
            self.assertEqual(metrics.rel_error_linf_expected,
                             rel_error_linf_expected)
            self.assertEqual(metrics.rel_error_expected, rel_error_expected)
            self.assertAlmostEqual(metrics.rel_error_l0_variance,
                                   rel_error_l0_variance,
                                   delta=1e-5)
            self.assertAlmostEqual(metrics.rel_error_variance,
                                   rel_error_variance,
                                   delta=1e-5)

        check_metric(col[0][0].privacy_id_count_metrics)
        check_metric(col[0][0].count_metrics)

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
        # Assert a singleton is returned
        self.assertLen(output, 1)
        # Assert there are 2 AggregateMetrics returned
        self.assertLen(output[0], 2)
        # Assert error_expected is correct.
        self.assertAlmostEqual(output[0][0].count_metrics.error_expected, -1)
        self.assertAlmostEqual(output[0][1].count_metrics.error_expected, 0)


if __name__ == '__main__':
    absltest.main()
