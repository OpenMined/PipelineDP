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
                kept_partitions=metrics.MeanVariance(mean=1.0,
                                                     var=0.648377588998337)),
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
                            l0=metrics.MeanVariance(mean=-51.191276314622854,
                                                    var=10.238255262924572),
                            linf_min=0.0,
                            linf_max=-28.439597952568246),
                        mean=-79.6308742671911,
                        variance=15.661039914487562,
                        rmse=79.91004632976953,
                        l1=0.0,
                        rmse_with_dropped_partitions=83.41695701143286,
                        l1_with_dropped_partitions=0.0),
                    relative_error=metrics.ValueErrors(
                        bounding_errors=metrics.ContributionBoundingErrors(
                            l0=metrics.MeanVariance(mean=-1.7063758771540947,
                                                    var=0.011375839181027294),
                            linf_min=0.0,
                            linf_max=-0.9479865984189416),
                        mean=-2.6543624755730364,
                        variance=0.017401155460541728,
                        rmse=2.6636682109923178,
                        l1=0.0,
                        rmse_with_dropped_partitions=2.7805652337144293,
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
                            l0=metrics.MeanVariance(mean=-25.595638157311427,
                                                    var=2.559563815731143),
                            linf_min=0.0,
                            linf_max=0.0),
                        mean=-25.595638157311427,
                        variance=3.9152599786218905,
                        rmse=25.81223614193849,
                        l1=0.0,
                        rmse_with_dropped_partitions=27.515758658140925,
                        l1_with_dropped_partitions=0.0),
                    relative_error=metrics.ValueErrors(
                        bounding_errors=metrics.ContributionBoundingErrors(
                            l0=metrics.MeanVariance(mean=-2.559563815731143,
                                                    var=0.025595638157311418),
                            linf_min=0.0,
                            linf_max=0.0),
                        mean=-2.559563815731143,
                        variance=0.03915259978621889,
                        rmse=2.5812236141938487,
                        l1=0.0,
                        rmse_with_dropped_partitions=2.751575865814092,
                        l1_with_dropped_partitions=0.0))
            ])
        common.assert_dataclasses_are_equal(self, report, expected)

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

        col = analysis.perform_utility_analysis(
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


if __name__ == '__main__':
    absltest.main()
