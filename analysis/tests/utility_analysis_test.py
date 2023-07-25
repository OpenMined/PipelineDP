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
import copy

from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
import analysis
from analysis import metrics
from analysis import utility_analysis
from analysis.tests import common


class UtilityAnalysis(parameterized.TestCase):

    def _get_default_extractors(self) -> pipeline_dp.DataExtractors:
        return pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    def _get_sum_metrics(self, sum_value: float) -> metrics.SumMetrics:
        return metrics.SumMetrics(aggregation=pipeline_dp.Metrics.COUNT,
                                  sum=sum_value,
                                  clipping_to_max_error=1,
                                  clipping_to_min_error=0,
                                  expected_l0_bounding_error=0,
                                  std_l0_bounding_error=0,
                                  std_noise=1,
                                  noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)

    def _get_per_partition_metrics(self, n_configurations=3):
        result = []
        for i in range(n_configurations):
            result.append(
                metrics.PerPartitionMetrics(
                    partition_selection_probability_to_keep=0.1,
                    statistics=metrics.Statistics(privacy_id_count=5, count=10),
                    metric_errors=[self._get_sum_metrics(150)]))
        return result

    @parameterized.parameters(False, True)
    def test_wo_public_partitions(self, pre_aggregated: bool):
        # Arrange
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[
                pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT
            ],
            max_partitions_contributed=1,
            max_contributions_per_partition=2)

        # Input collection has 10 privacy ids where each privacy id
        # contributes to the same 10 partitions, three times in each partition.
        if not pre_aggregated:
            col = [(i, j) for i in range(10) for j in range(10)] * 3
        else:
            # This is a pre-aggregated dataset, namely each element has a format
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
            data_extractors = pipeline_dp.PreAggregateExtractors(
                partition_extractor=lambda x: f"pk{x[0]}",
                preaggregate_extractor=lambda x: x[1])

        col, per_partition_result = analysis.perform_utility_analysis(
            col=col,
            backend=pipeline_dp.LocalBackend(),
            options=analysis.UtilityAnalysisOptions(
                epsilon=3,
                delta=0.9,
                aggregate_params=aggregate_params,
                pre_aggregated_data=pre_aggregated),
            data_extractors=data_extractors)

        col = list(col)
        per_partition_result = list(per_partition_result)

        # Assert
        self.assertLen(col, 1)
        report = col[0]
        self.assertIsInstance(report, metrics.UtilityReport)
        expected = metrics.UtilityReport(
            configuration_index=0,
            partitions_info=metrics.PartitionsInfo(
                public_partitions=False,
                num_dataset_partitions=10,
                num_non_public_partitions=None,
                num_empty_partitions=None,
                strategy=None,
                kept_partitions=metrics.MeanVariance(mean=3.51622411,
                                                     var=2.2798409)),
            metric_errors=[
                metrics.MetricUtility(
                    metric=pipeline_dp.Metrics.COUNT,
                    noise_std=1.380859375,
                    noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                    ratio_data_dropped=metrics.DataDropInfo(
                        l0=0.6,
                        linf=0.333333333,
                        partition_selection=0.04322517259988915),
                    absolute_error=metrics.ValueErrors(
                        bounding_errors=metrics.ContributionBoundingErrors(
                            l0=metrics.MeanVariance(mean=-18, var=3.6),
                            linf_min=0.0,
                            linf_max=-10),
                        mean=-28,
                        variance=5.5067726,
                        rmse=28.098163153,
                        l1=0.0,
                        rmse_with_dropped_partitions=29.331271542782087,
                        l1_with_dropped_partitions=0.0),
                    relative_error=metrics.ValueErrors(
                        bounding_errors=metrics.ContributionBoundingErrors(
                            l0=metrics.MeanVariance(mean=-0.6, var=0.004),
                            linf_min=0.0,
                            linf_max=-0.33333333),
                        mean=-0.93333333,
                        variance=0.006118636237250433,
                        rmse=0.9366054384576044,
                        l1=0.0,
                        rmse_with_dropped_partitions=0.9777090514260699,
                        l1_with_dropped_partitions=0.0)),
                metrics.MetricUtility(
                    metric=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                    noise_std=0.6904296875,
                    noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                    ratio_data_dropped=metrics.DataDropInfo(
                        l0=0.9,
                        linf=-0.0,
                        partition_selection=0.06483775889983372),
                    absolute_error=metrics.ValueErrors(
                        bounding_errors=metrics.ContributionBoundingErrors(
                            l0=metrics.MeanVariance(mean=-9, var=0.9),
                            linf_min=0.0,
                            linf_max=0.0),
                        mean=-9,
                        variance=1.37669315,
                        rmse=9.07616070,
                        l1=0.0,
                        rmse_with_dropped_partitions=9.67515739991,
                        l1_with_dropped_partitions=0.0),
                    relative_error=metrics.ValueErrors(
                        bounding_errors=metrics.ContributionBoundingErrors(
                            l0=metrics.MeanVariance(mean=-0.9, var=0.009),
                            linf_min=0.0,
                            linf_max=0.0),
                        mean=-0.9,
                        variance=0.013766931533,
                        rmse=0.90761607055726,
                        l1=0.0,
                        rmse_with_dropped_partitions=0.9675157399915,
                        l1_with_dropped_partitions=0.0))
            ])
        expected_copy = copy.deepcopy(expected)
        expected.utility_report_histogram = [
            metrics.UtilityReportBin(partition_size_from=20,
                                     partition_size_to=50,
                                     report=expected_copy)
        ]
        common.assert_dataclasses_are_equal(self, report, expected)
        self.assertLen(per_partition_result, 10)

    @parameterized.named_parameters(
        dict(testcase_name="Gaussian noise",
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             expected_noise_std=5.9765625),
        dict(testcase_name="Laplace noise",
             noise_kind=pipeline_dp.NoiseKind.LAPLACE,
             expected_noise_std=1.4142135623730951),
    )
    def test_w_public_partitions(self, noise_kind, expected_noise_std):
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

        col, _ = analysis.perform_utility_analysis(
            col=col,
            backend=pipeline_dp.LocalBackend(),
            options=analysis.UtilityAnalysisOptions(
                epsilon=2, delta=1e-10, aggregate_params=aggregator_params),
            data_extractors=data_extractor,
            public_partitions=public_partitions)

        col = list(col)

        # Assert
        self.assertLen(col, 1)
        report: metrics.UtilityReport = col[0]

        self.assertLen(report.metric_errors, 2)  # COUNT and PRIVACY_ID_COUNT
        errors = report.metric_errors
        [self.assertEqual(e.noise_kind, noise_kind) for e in errors]
        [self.assertEqual(e.noise_std, expected_noise_std) for e in errors]

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

        output, _ = analysis.perform_utility_analysis(
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

        utility_reports = list(output)

        # Assert
        self.assertLen(utility_reports, 2)  # one report per each configuration.

        # Check the parameter configuration
        expected_noise_std = [3.02734375, 8.56262117843085]
        expected_l0_error = [-0.5, 0]
        expected_partition_info = metrics.PartitionsInfo(
            public_partitions=True,
            num_dataset_partitions=2,
            num_non_public_partitions=0,
            num_empty_partitions=0)
        for i_configuration, report in enumerate(utility_reports):
            self.assertEqual(report.configuration_index, i_configuration)
            self.assertEqual(report.partitions_info, expected_partition_info)
            self.assertLen(report.metric_errors, 1)  # metrics for COUNT
            errors = report.metric_errors[0]
            self.assertEqual(errors.metric, pipeline_dp.Metrics.COUNT)
            self.assertEqual(errors.noise_std,
                             expected_noise_std[i_configuration])
            self.assertEqual(errors.absolute_error.bounding_errors.l0.mean,
                             expected_l0_error[i_configuration])

    def test_generate_bucket_bounds(self):
        self.assertLen(utility_analysis._generate_bucket_bounds(), 29)
        self.assertEqual(utility_analysis._generate_bucket_bounds()[:10],
                         (0, 1, 10, 20, 50, 100, 200, 500, 1000, 2000))

    def test_get_lower_bound(self):
        self.assertEqual(utility_analysis._get_lower_bound(-1), 0)
        self.assertEqual(utility_analysis._get_lower_bound(0), 0)
        self.assertEqual(utility_analysis._get_lower_bound(1), 1)
        self.assertEqual(utility_analysis._get_lower_bound(5), 1)
        self.assertEqual(utility_analysis._get_lower_bound(11), 10)
        self.assertEqual(utility_analysis._get_lower_bound(20), 20)
        self.assertEqual(utility_analysis._get_lower_bound(1234), 1000)

    def test_get_upper_bound(self):
        self.assertEqual(utility_analysis._get_upper_bound(-1), 0)
        self.assertEqual(utility_analysis._get_upper_bound(0), 1)
        self.assertEqual(utility_analysis._get_upper_bound(1), 10)
        self.assertEqual(utility_analysis._get_upper_bound(5), 10)
        self.assertEqual(utility_analysis._get_upper_bound(11), 20)
        self.assertEqual(utility_analysis._get_upper_bound(20), 50)
        self.assertEqual(utility_analysis._get_upper_bound(1234), 2000)

    def test_unnest_metrics(self):
        input_data = self._get_per_partition_metrics(n_configurations=2)
        output = list(utility_analysis._unnest_metrics(input_data))
        self.assertLen(output, 4)
        self.assertEqual(output[0], ((0, None), input_data[0]))
        self.assertEqual(output[1], ((0, 100), input_data[0]))
        self.assertEqual(output[2], ((1, None), input_data[1]))
        self.assertEqual(output[3], ((1, 100), input_data[1]))


if __name__ == '__main__':
    absltest.main()
