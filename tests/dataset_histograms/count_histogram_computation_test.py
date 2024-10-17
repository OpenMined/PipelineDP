# Copyright 2024 OpenMined.
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
"""Tests for computing count dataset histograms."""

from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import count_histogram_computation
from analysis import pre_aggregation


class CountHistogramsTest(parameterized.TestCase):

    def test_to_bin_lower_upper_logarithmic(self):
        to_bin_lower = count_histogram_computation._to_bin_lower_upper_logarithmic
        self.assertEqual(to_bin_lower(1), (1, 2))
        self.assertEqual(to_bin_lower(999), (999, 1000))
        self.assertEqual(to_bin_lower(1000), (1000, 1010))
        self.assertEqual(to_bin_lower(1001), (1000, 1010))
        self.assertEqual(to_bin_lower(1012), (1010, 1020))
        self.assertEqual(to_bin_lower(2022), (2020, 2030))
        self.assertEqual(to_bin_lower(12522), (12500, 12600))
        self.assertEqual(to_bin_lower(10**9 + 10**7 + 1234),
                         (10**9 + 10**7, 10**9 + 2 * 10**7))

    @parameterized.named_parameters(
        dict(testcase_name='empty', input=[], expected=[]),
        dict(testcase_name='small_histogram',
             input=[3, 3, 1, 1, 2, 10],
             expected=[
                 hist.FrequencyBin(lower=1,
                                   upper=2,
                                   count=2,
                                   sum=2,
                                   min=1,
                                   max=1),
                 hist.FrequencyBin(lower=2,
                                   upper=3,
                                   count=1,
                                   sum=2,
                                   min=1,
                                   max=2),
                 hist.FrequencyBin(lower=3,
                                   upper=4,
                                   count=2,
                                   sum=6,
                                   min=1,
                                   max=3),
                 hist.FrequencyBin(lower=10,
                                   upper=11,
                                   count=1,
                                   sum=10,
                                   min=1,
                                   max=10)
             ]),
        dict(testcase_name='histogram_with_bins_wider_1',
             input=[1005, 3, 12345, 12346],
             expected=[
                 hist.FrequencyBin(lower=3,
                                   upper=4,
                                   count=1,
                                   sum=3,
                                   min=1,
                                   max=3),
                 hist.FrequencyBin(lower=1000,
                                   upper=1010,
                                   count=1,
                                   sum=1005,
                                   min=1,
                                   max=1005),
                 hist.FrequencyBin(lower=12300,
                                   upper=12400,
                                   count=2,
                                   sum=24691,
                                   min=1,
                                   max=12346)
             ]),
    )
    def test_compute_frequency_histogram(self, input, expected):
        backend = pipeline_dp.LocalBackend()
        histogram = count_histogram_computation._compute_frequency_histogram(
            input, backend, "histogram_name")
        histogram = list(histogram)
        self.assertLen(histogram, 1)
        histogram = histogram[0]

        self.assertEqual("histogram_name", histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1, sum=1, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=1, sum=2, min=1, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, min=1, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1, sum=1, min=1, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1230,
                                      upper=1240,
                                      count=1,
                                      sum=1234,
                                      min=1,
                                      max=1234),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, min=1, max=15),
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_l0_contributions_histogram(self, testcase_name, input,
                                                expected, pre_aggregated):
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(
                input,
                backend,
                data_extractors=pipeline_dp.DataExtractors(
                    privacy_id_extractor=lambda x: x[0],
                    partition_extractor=lambda x: x[1],
                    value_extractor=lambda x: 0))
            compute_histograms = count_histogram_computation._compute_l0_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = count_histogram_computation._compute_l0_contributions_histogram
        histogram = list(compute_histograms(input, backend))[0]
        self.assertEqual(hist.HistogramType.L0_CONTRIBUTIONS, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1, sum=1, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=1, sum=2, min=1, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, min=1, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, min=1, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i // 2) for i in range(1235)
                      ],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1230,
                                      upper=1240,
                                      count=1,
                                      sum=1235,
                                      min=1,
                                      max=1235),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, min=1, max=15),
                ]),
            dict(
                testcase_name='3 privacy ids',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)] +
                [(2, i) for i in range(11)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=11, upper=12, count=1, sum=11, min=1, max=11),
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, min=1, max=15),
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_l1_contributions_histogram(self, testcase_name, input,
                                                expected, pre_aggregated):
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(
                input,
                backend,
                data_extractors=pipeline_dp.DataExtractors(
                    privacy_id_extractor=lambda x: x[0],
                    partition_extractor=lambda x: x[1],
                    value_extractor=lambda x: 0))
            compute_histograms = count_histogram_computation._compute_l1_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = count_histogram_computation._compute_l1_contributions_histogram
        histogram = list(compute_histograms(input, backend))[0]
        self.assertEqual(hist.HistogramType.L1_CONTRIBUTIONS, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1),
                       (1, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=2, sum=2, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=1, sum=2, min=1, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, min=1, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, min=1, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, min=1, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=30, sum=30, min=1, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids',
                input=[(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 0),
                       (1, 2)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=2, sum=2, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=1, sum=2, min=1, max=2),
                    hist.FrequencyBin(
                        lower=3, upper=4, count=1, sum=3, min=1, max=3),
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_linf_contributions_histogram(self, testcase_name, input,
                                                  expected, pre_aggregated):
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(
                input,
                backend,
                data_extractors=pipeline_dp.DataExtractors(
                    privacy_id_extractor=lambda x: x[0],
                    partition_extractor=lambda x: x[1],
                    value_extractor=lambda x: 0))
            compute_histograms = count_histogram_computation._compute_linf_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = count_histogram_computation._compute_linf_contributions_histogram
        histogram = list(compute_histograms(input, backend))
        self.assertLen(histogram, 1)
        histogram = histogram[0]
        self.assertEqual(hist.HistogramType.LINF_CONTRIBUTIONS, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty histogram', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1),
                       (1, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1, sum=1, min=1, max=1),
                    hist.FrequencyBin(
                        lower=3, upper=3, count=1, sum=3, min=1, max=3)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, min=1, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, min=1, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partitions',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, min=1, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=20, sum=20, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=5, sum=10, min=1, max=2),
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_partitions_count_histogram(self, testcase_name, input,
                                                expected, pre_aggregated):
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(
                input,
                backend,
                data_extractors=pipeline_dp.DataExtractors(
                    privacy_id_extractor=lambda x: x[0],
                    partition_extractor=lambda x: x[1],
                    value_extractor=lambda x: 0))
            compute_histograms = count_histogram_computation._compute_partition_count_histogram_on_preaggregated_data
        else:
            compute_histograms = count_histogram_computation._compute_partition_count_histogram
        histogram = list(compute_histograms(input, backend))[0]
        self.assertEqual(hist.HistogramType.COUNT_PER_PARTITION, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty histogram', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1, sum=1, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=1, sum=2, min=1, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, min=1, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1, sum=1, min=1, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partitions',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, min=1, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=20, sum=20, min=1, max=1),
                    hist.FrequencyBin(
                        lower=2, upper=3, count=5, sum=10, min=1, max=2),
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_partitions_privacy_id_count_histogram(
            self, testcase_name, input, expected, pre_aggregated):
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(
                input,
                backend,
                data_extractors=pipeline_dp.DataExtractors(
                    privacy_id_extractor=lambda x: x[0],
                    partition_extractor=lambda x: x[1],
                    value_extractor=lambda x: 0))
            compute_histograms = count_histogram_computation._compute_partition_privacy_id_count_histogram_on_preaggregated_data
        else:
            compute_histograms = count_histogram_computation._compute_partition_privacy_id_count_histogram

        histogram = list(compute_histograms(input, backend))[0]
        self.assertEqual(hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION,
                         histogram.name)
        self.assertListEqual(expected, histogram.bins)


if __name__ == '__main__':
    absltest.main()
