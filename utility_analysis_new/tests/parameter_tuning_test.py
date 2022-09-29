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
"""Parameter tuning test."""

from absl.testing import absltest
from absl.testing import parameterized
from unittest import mock
from typing import List

import pipeline_dp
from utility_analysis_new import combiners
from utility_analysis_new import parameter_tuning
from utility_analysis_new.parameter_tuning import FrequencyBin


def _get_aggregate_params():
    # Limit contributions to 1 per partition, contribution error will be half of the count.
    return pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1)


class ParameterTuning(parameterized.TestCase):

    def test_to_bin_lower(self):
        to_bin_lower = parameter_tuning._to_bin_lower
        self.assertEqual(to_bin_lower(1), 1)
        self.assertEqual(to_bin_lower(999), 999)
        self.assertEqual(to_bin_lower(1000), 1000)
        self.assertEqual(to_bin_lower(1001), 1000)
        self.assertEqual(to_bin_lower(1012), 1010)
        self.assertEqual(to_bin_lower(2022), 2020)
        self.assertEqual(to_bin_lower(12522), 12500)
        self.assertEqual(to_bin_lower(10**9 + 10**7 + 1234), 10**9 + 10**7)

    @parameterized.named_parameters(
        dict(testcase_name='empty', input=[], expected=[]),
        dict(testcase_name='small_histogram',
             input=[3, 3, 1, 1, 2, 10],
             expected=[
                 FrequencyBin(lower=1, count=2, sum=2, max=1),
                 FrequencyBin(lower=2, count=1, sum=2, max=2),
                 FrequencyBin(lower=3, count=2, sum=6, max=3),
                 FrequencyBin(lower=10, count=1, sum=10, max=10)
             ]),
        dict(testcase_name='histogram_with_bins_wider_1',
             input=[1005, 3, 12345, 12346],
             expected=[
                 FrequencyBin(lower=3, count=1, sum=3, max=3),
                 FrequencyBin(lower=1000, count=1, sum=1005, max=1005),
                 FrequencyBin(lower=12300, count=2, sum=24691, max=12346)
             ]),
    )
    def test_compute_frequency_histogram(self, input, expected):
        backend = pipeline_dp.LocalBackend()
        histogram = parameter_tuning._compute_frequency_histogram(
            input, backend, "histogram_name")
        histogram = list(histogram)
        self.assertLen(histogram, 1)
        histogram = histogram[0]

        self.assertEqual("histogram_name", histogram.name)
        self.assertListEqual(expected, histogram.bins)

    def test_list_to_contribution_histograms(self):
        histogram1 = parameter_tuning.Histogram("CrossPartitionHistogram", None)
        histogram2 = parameter_tuning.Histogram("PerPartitionHistogram", None)
        histograms = parameter_tuning._list_to_contribution_histograms(
            [histogram2, histogram1])
        self.assertEqual(histogram1, histograms.cross_partition_histogram)
        self.assertEqual(histogram2, histograms.per_partition_histogram)

    @parameterized.named_parameters(
        dict(testcase_name='empty', input=[], expected=[]),
        dict(
            testcase_name='small_histogram',
            input=[(1, 1), (1, 2), (2, 1), (1, 1)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=1, sum=1, max=1),
                FrequencyBin(lower=2, count=1, sum=2, max=2)
            ]),
        dict(
            testcase_name='Each privacy id, 1 contribution',
            input=[(i, i) for i in range(100)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=100, sum=100, max=1),
            ]),
        dict(
            testcase_name='1 privacy id many contributions to 1 partition',
            input=[(0, 0)] * 100,  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=1, sum=1, max=1),
            ]),
        dict(
            testcase_name='1 privacy id many contributions to many partition',
            input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1230, count=1, sum=1234, max=1234),
            ]),
        dict(
            testcase_name='2 privacy ids, same partitions contributed',
            input=[(0, i) for i in range(15)] +
            [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=15, count=2, sum=30, max=15),
            ]),
    )
    def test_compute_cross_partition_histogram(self, input, expected):
        histogram = parameter_tuning._compute_cross_partition_histogram(
            input, pipeline_dp.LocalBackend())
        histogram = list(histogram)[0]
        self.assertEqual("CrossPartitionHistogram", histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.named_parameters(
        dict(testcase_name='empty', input=[], expected=[]),
        dict(
            testcase_name='small_histogram',
            input=[(1, 1), (1, 2), (2, 1), (1, 1)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=2, sum=2, max=1),
                FrequencyBin(lower=2, count=1, sum=2, max=2)
            ]),
        dict(
            testcase_name='Each privacy id, 1 contribution',
            input=[(i, i) for i in range(100)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=100, sum=100, max=1),
            ]),
        dict(
            testcase_name='1 privacy id many contributions to 1 partition',
            input=[(0, 0)] * 100,  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=100, count=1, sum=100, max=100),
            ]),
        dict(
            testcase_name='1 privacy id many contributions to many partition',
            input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=1234, sum=1234, max=1),
            ]),
        dict(
            testcase_name='2 privacy ids, same partitions contributed',
            input=[(0, i) for i in range(15)] +
            [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=30, sum=30, max=1),
            ]),
        dict(
            testcase_name='2 privacy ids',
            input=[(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 0),
                   (1, 2)],  # (privacy_id, partition)
            expected=[
                FrequencyBin(lower=1, count=2, sum=2, max=1),
                FrequencyBin(lower=2, count=1, sum=2, max=2),
                FrequencyBin(lower=3, count=1, sum=3, max=3),
            ]),
    )
    def test_compute_per_partition_histogram(self, input, expected):
        histogram = parameter_tuning._compute_per_partition_histogram(
            input, pipeline_dp.LocalBackend())

        histogram = list(histogram)
        self.assertLen(histogram, 1)
        histogram = histogram[0]
        self.assertEqual("PerPartitionHistogram", histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.named_parameters(
        dict(testcase_name='empty',
             input=[],
             expected_cross_partition=[],
             expected_per_partition=[]),
        dict(
            testcase_name='small_histogram',
            input=[(1, 1), (1, 2), (2, 1), (1, 1)],  # (privacy_id, partition)
            expected_cross_partition=[
                FrequencyBin(lower=1, count=1, sum=1, max=1),
                FrequencyBin(lower=2, count=1, sum=2, max=2)
            ],
            expected_per_partition=[
                FrequencyBin(lower=1, count=2, sum=2, max=1),
                FrequencyBin(lower=2, count=1, sum=2, max=2)
            ]),
        dict(
            testcase_name='Each privacy id, 1 contribution',
            input=[(i, i) for i in range(100)],  # (privacy_id, partition)
            expected_cross_partition=[
                FrequencyBin(lower=1, count=100, sum=100, max=1),
            ],
            expected_per_partition=[
                FrequencyBin(lower=1, count=100, sum=100, max=1),
            ]),
        dict(
            testcase_name='1 privacy id many contributions to 1 partition',
            input=[(0, 0)] * 100,  # (privacy_id, partition)
            expected_cross_partition=[
                FrequencyBin(lower=1, count=1, sum=1, max=1),
            ],
            expected_per_partition=[
                FrequencyBin(lower=100, count=1, sum=100, max=100),
            ]),
        dict(
            testcase_name='1 privacy id many contributions to many partition',
            input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
            expected_cross_partition=[
                FrequencyBin(lower=1230, count=1, sum=1234, max=1234),
            ],
            expected_per_partition=[
                FrequencyBin(lower=1, count=1234, sum=1234, max=1),
            ]),
        dict(
            testcase_name='2 privacy ids, same partitions contributed',
            input=[(0, i) for i in range(15)] +
            [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
            expected_cross_partition=[
                FrequencyBin(lower=15, count=2, sum=30, max=15),
            ],
            expected_per_partition=[
                FrequencyBin(lower=1, count=30, sum=30, max=1),
            ]),
        dict(
            testcase_name='2 privacy ids',
            input=[(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 0),
                   (1, 2)],  # (privacy_id, partition)
            expected_cross_partition=[
                FrequencyBin(lower=2, count=2, sum=4, max=2),
            ],
            expected_per_partition=[
                FrequencyBin(lower=1, count=2, sum=2, max=1),
                FrequencyBin(lower=2, count=1, sum=2, max=2),
                FrequencyBin(lower=3, count=1, sum=3, max=3),
            ]),
    )
    def test_compute_contribution_histograms(self, input,
                                             expected_cross_partition,
                                             expected_per_partition):
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
        )
        histograms = parameter_tuning.compute_contribution_histograms(
            input, data_extractors, pipeline_dp.LocalBackend())
        histograms = list(histograms)
        self.assertLen(histograms, 1)
        histograms = histograms[0]

        self.assertEqual("CrossPartitionHistogram",
                         histograms.cross_partition_histogram.name)
        self.assertListEqual(expected_cross_partition,
                             histograms.cross_partition_histogram.bins)
        self.assertEqual("PerPartitionHistogram",
                         histograms.per_partition_histogram.name)
        self.assertListEqual(expected_per_partition,
                             histograms.per_partition_histogram.bins)

    @parameterized.named_parameters(
        dict(testcase_name='1 bins histogram',
             bins=[
                 FrequencyBin(lower=1000, count=10, sum=10100, max=1009),
             ],
             q=[0.05, 0.1, 0.5, 0.8, 0.9],
             expected_quantiles=[1000, 1000, 1000, 1000, 1000]),
        dict(testcase_name='6 bins histogram',
             bins=[
                 FrequencyBin(lower=1, count=2, sum=2, max=1),
                 FrequencyBin(lower=2, count=1, sum=2, max=2),
                 FrequencyBin(lower=3, count=1, sum=3, max=3),
                 FrequencyBin(lower=4, count=2, sum=8, max=4),
                 FrequencyBin(lower=5, count=2, sum=10, max=5),
                 FrequencyBin(lower=6, count=1, sum=6, max=6),
                 FrequencyBin(lower=10, count=1, sum=11, max=11)
             ],
             q=[0.001, 0.05, 0.1, 0.5, 0.8, 0.9],
             expected_quantiles=[1, 1, 1, 4, 6, 10]))
    def test_quantile_contributions(self, bins, q, expected_quantiles):
        histogram = parameter_tuning.Histogram("name", bins)
        output = histogram.quantiles(q)
        self.assertListEqual(expected_quantiles, output)

    @parameterized.parameters(
        (True, True, [1, 1, 2, 2, 6, 6], [3, 6, 3, 6, 3, 6]),
        (False, True, None, [3, 6]), (True, False, [1, 2, 6], None))
    def test_find_candidate_parameters(
        self,
        tune_max_partitions_contributed: bool,
        tune_max_contributions_per_partition: bool,
        expected_max_partitions_contributed: List,
        expected_max_contributions_per_partition: List,
    ):
        mock_l0_histogram = parameter_tuning.Histogram(None, None)
        mock_l0_histogram.quantiles = mock.Mock(return_value=[1, 1, 2])
        setattr(mock_l0_histogram.__class__, 'max_value', 6)
        mock_linf_histogram = parameter_tuning.Histogram(None, None)
        mock_linf_histogram.quantiles = mock.Mock(return_value=[3, 6, 6])

        mock_histograms = parameter_tuning.ContributionHistograms(
            mock_l0_histogram, mock_linf_histogram)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=tune_max_partitions_contributed,
            max_contributions_per_partition=tune_max_contributions_per_partition
        )

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms, parameters_to_tune)
        self.assertEqual(expected_max_partitions_contributed,
                         candidates.max_partitions_contributed)
        self.assertEqual(expected_max_contributions_per_partition,
                         candidates.max_contributions_per_partition)

    def test_tune(self):
        input = [(i % 10, f"pk{i/10}") for i in range(10)]
        public_partitions = [f"pk{i}" for i in range(10)]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: None)

        contribution_histograms = list(
            parameter_tuning.compute_contribution_histograms(
                input, data_extractors, pipeline_dp.LocalBackend()))[0]

        tune_options = parameter_tuning.TuneOptions(
            epsilon=1,
            delta=1e-10,
            aggregate_params=_get_aggregate_params(),
            function_to_minimize=parameter_tuning.MinimizingFunction.
            ABSOLUTE_ERROR,
            parameters_to_tune=parameter_tuning.ParametersToTune(True, True))
        tune_result = list(
            parameter_tuning.tune(input, pipeline_dp.LocalBackend(),
                                  contribution_histograms, tune_options,
                                  data_extractors, public_partitions))[0]

        self.assertEqual(tune_options, tune_result.options)
        self.assertEqual(contribution_histograms,
                         tune_result.contribution_histograms)
        self.assertIsInstance(tune_result.utility_analysis_results[0],
                              combiners.AggregateErrorMetrics)


if __name__ == '__main__':
    absltest.main()
