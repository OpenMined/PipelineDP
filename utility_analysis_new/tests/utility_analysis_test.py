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
import utility_analysis_new
from utility_analysis_new import utility_analysis


class UtilityAnalysis(parameterized.TestCase):

    def _get_default_extractors(self) -> pipeline_dp.DataExtractors:
        return pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    def test_wo_public_partitions(self):
        # Arrange
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=2)

        # Input collection has 10 privacy ids where each privacy id
        # contributes to the same 10 partitions, three times in each partition.
        col = [(i, j) for i in range(10) for j in range(10)] * 3
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: f"pk{x[1]}",
            value_extractor=lambda x: None)

        col = utility_analysis.perform_utility_analysis(
            col=col,
            backend=pipeline_dp.LocalBackend(),
            options=utility_analysis.UtilityAnalysisOptions(
                eps=2, delta=0.9, aggregate_params=aggregate_params),
            data_extractors=data_extractors)

        col = list(col)

        # Assert
        # Assert a singleton is returned
        self.assertLen(col, 1)
        # Assert there are 2 AggregateErrorMetrics, one for private partition
        # selection and 1 for count.
        self.assertEqual(len(col[0]), 2)
        # Assert partition selection metrics are reasonable.
        # partition_kept_probability = 0.4311114 for each partition
        self.assertEqual(col[0][0].num_partitions, 10)
        self.assertAlmostEqual(col[0][0].dropped_partitions_expected,
                               5.68885,
                               delta=1e-5)
        self.assertAlmostEqual(col[0][0].dropped_partitions_variance,
                               2.45254,
                               delta=1e-5)
        # Assert count metrics are reasonable.
        self.assertAlmostEqual(col[0][1].abs_error_expected,
                               -12.07119,
                               delta=1e-5)
        self.assertAlmostEqual(col[0][1].abs_error_variance,
                               2.06498,
                               delta=1e-5)
        self.assertAlmostEqual(col[0][1].rel_error_expected,
                               -0.40237,
                               delta=1e-5)
        self.assertAlmostEqual(col[0][1].rel_error_variance,
                               0.002294,
                               delta=1e-5)

    def test_w_public_partitions(self):
        # Arrange
        aggregator_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT],
            max_partitions_contributed=1,
            max_contributions_per_partition=1)

        public_partitions = ["pk0", "pk1", "pk101"]

        # Input collection has 100 elements, such that each privacy id
        # contributes 1 time and each partition has 1 element.
        col = list(range(100))
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: f"pk{x}",
            value_extractor=lambda x: None)

        col = utility_analysis.perform_utility_analysis(
            col=col,
            backend=pipeline_dp.LocalBackend(),
            options=utility_analysis.UtilityAnalysisOptions(
                eps=2, delta=1e-10, aggregate_params=aggregator_params),
            data_extractors=data_extractor,
            public_partitions=public_partitions)

        col = list(col)

        # Simply assert pipeline can run for now.
        # Assert
        # Assert a singleton is returned
        self.assertLen(col, 1)
        # Assert there is only a single AggregateErrorMetrics & no metrics for
        # partition selection are returned.
        self.assertLen(col[0], 1)
        # Assert count metrics are reasonable.
        # Relative errors are infinity due to the empty partition.
        self.assertAlmostEqual(col[0][0].abs_error_expected, 0, delta=1e-2)
        self.assertAlmostEqual(col[0][0].abs_error_variance, 9.1648, delta=1e-2)
        self.assertEqual(col[0][0].rel_error_expected, float('inf'))
        self.assertEqual(col[0][0].rel_error_variance, float('inf'))

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

        # Input collection has 1 privacy id, which contributes to 2 partitions
        # 1 and 2 times correspondingly.
        input = [(0, "pk0"), (0, "pk1"), (0, "pk1")]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: None)

        public_partitions = ["pk0", "pk1"]

        output = utility_analysis.perform_utility_analysis(
            col=input,
            backend=pipeline_dp.LocalBackend(),
            options=utility_analysis.UtilityAnalysisOptions(
                eps=2,
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
        # Assert there are 2 AggregateErrorMetrics returned
        self.assertLen(output[0], 2)
        # Assert abs_error_expected is correct.
        self.assertAlmostEqual(output[0][0].abs_error_expected, -1)
        self.assertAlmostEqual(output[0][1].abs_error_expected, 0)


if __name__ == '__main__':
    absltest.main()
