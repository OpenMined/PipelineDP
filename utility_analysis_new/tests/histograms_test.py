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

from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from utility_analysis_new import histograms as hist
from utility_analysis_new.histograms import FrequencyBin


class ParameterTuning(parameterized.TestCase):

    def test_to_bin_lower(self):
        to_bin_lower = hist._to_bin_lower
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
        histogram = hist._compute_frequency_histogram(input, backend,
                                                      "histogram_name")
        histogram = list(histogram)
        self.assertLen(histogram, 1)
        histogram = histogram[0]

        self.assertEqual("histogram_name", histogram.name)
        self.assertListEqual(expected, histogram.bins)

    def test_list_to_contribution_histograms(self):
        histogram1 = hist.Histogram("CrossPartitionHistogram", None)
        histogram2 = hist.Histogram("PerPartitionHistogram", None)
        histograms = hist._list_to_contribution_histograms(
            [histogram2, histogram1])
        self.assertEqual(histogram1, histograms.l0_contributions_histogram)
        self.assertEqual(histogram2, histograms.linf_contributions_histogram)

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
        histogram = hist._compute_cross_partition_histogram(
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
        histogram = hist._compute_per_partition_histogram(
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
        histograms = hist.compute_contribution_histograms(
            input, data_extractors, pipeline_dp.LocalBackend())
        histograms = list(histograms)
        self.assertLen(histograms, 1)
        histograms = histograms[0]

        self.assertEqual("CrossPartitionHistogram",
                         histograms.l0_contributions_histogram.name)
        self.assertListEqual(expected_cross_partition,
                             histograms.l0_contributions_histogram.bins)
        self.assertEqual("PerPartitionHistogram",
                         histograms.linf_contributions_histogram.name)
        self.assertListEqual(expected_per_partition,
                             histograms.linf_contributions_histogram.bins)

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
        histogram = hist.Histogram("name", bins)
        output = histogram.quantiles(q)
        self.assertListEqual(expected_quantiles, output)


if __name__ == '__main__':
    absltest.main()
