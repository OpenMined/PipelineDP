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
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import computing_histograms
from analysis import pre_aggregation


class ComputingHistogramsTest(parameterized.TestCase):

    def test_to_bin_lower(self):
        to_bin_lower = computing_histograms._to_bin_lower
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
                 hist.FrequencyBin(lower=1, count=2, sum=2, max=1),
                 hist.FrequencyBin(lower=2, count=1, sum=2, max=2),
                 hist.FrequencyBin(lower=3, count=2, sum=6, max=3),
                 hist.FrequencyBin(lower=10, count=1, sum=10, max=10)
             ]),
        dict(testcase_name='histogram_with_bins_wider_1',
             input=[1005, 3, 12345, 12346],
             expected=[
                 hist.FrequencyBin(lower=3, count=1, sum=3, max=3),
                 hist.FrequencyBin(lower=1000, count=1, sum=1005, max=1005),
                 hist.FrequencyBin(lower=12300, count=2, sum=24691, max=12346)
             ]),
    )
    def test_compute_frequency_histogram(self, input, expected):
        backend = pipeline_dp.LocalBackend()
        histogram = computing_histograms._compute_frequency_histogram(
            input, backend, "histogram_name")
        histogram = list(histogram)
        self.assertLen(histogram, 1)
        histogram = histogram[0]

        self.assertEqual("histogram_name", histogram.name)
        self.assertListEqual(expected, histogram.bins)

    def test_list_to_contribution_histograms(self):
        histogram1 = hist.Histogram(hist.HistogramType.L0_CONTRIBUTIONS, None)
        histogram2 = hist.Histogram(hist.HistogramType.L1_CONTRIBUTIONS, None)
        histogram3 = hist.Histogram(hist.HistogramType.LINF_CONTRIBUTIONS, None)
        histogram4 = hist.Histogram(hist.HistogramType.COUNT_PER_PARTITION,
                                    None)
        histogram5 = hist.Histogram(
            hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION, None)
        histograms = computing_histograms._list_to_contribution_histograms(
            [histogram2, histogram1, histogram3, histogram5, histogram4])
        self.assertEqual(histogram1, histograms.l0_contributions_histogram)
        self.assertEqual(histogram2, histograms.l1_contributions_histogram)
        self.assertEqual(histogram3, histograms.linf_contributions_histogram)
        self.assertEqual(histogram4, histograms.count_per_partition_histogram)
        self.assertEqual(histogram5, histograms.count_privacy_id_per_partition)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1230, count=1, sum=1234, max=1234),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=15, count=2, sum=30, max=15),
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
            compute_histograms = computing_histograms._compute_l0_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_l0_contributions_histogram
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
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=100, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i // 2) for i in range(1235)
                      ],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1230, count=1, sum=1235, max=1235),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=15, count=2, sum=30, max=15),
                ]),
            dict(
                testcase_name='3 privacy ids',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)] +
                [(2, i) for i in range(11)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=11, count=1, sum=11, max=11),
                    hist.FrequencyBin(lower=15, count=2, sum=30, max=15),
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
            compute_histograms = computing_histograms._compute_l1_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_l1_contributions_histogram
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
                    hist.FrequencyBin(lower=1, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=100, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=30, sum=30, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids',
                input=[(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 0),
                       (1, 2)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2),
                    hist.FrequencyBin(lower=3, count=1, sum=3, max=3),
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
            compute_histograms = computing_histograms._compute_linf_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_linf_contributions_histogram
        histogram = list(compute_histograms(input, backend))
        self.assertLen(histogram, 1)
        histogram = histogram[0]
        self.assertEqual(hist.HistogramType.LINF_CONTRIBUTIONS, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1),
                       (1, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=3, count=1, sum=3, max=3)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=100, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partitions',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=20, sum=20, max=1),
                    hist.FrequencyBin(lower=2, count=5, sum=10, max=2),
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
            compute_histograms = computing_histograms._compute_partition_count_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_partition_count_histogram
        histogram = list(compute_histograms(input, backend))[0]
        self.assertEqual(hist.HistogramType.COUNT_PER_PARTITION, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partitions',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, count=20, sum=20, max=1),
                    hist.FrequencyBin(lower=2, count=5, sum=10, max=2),
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
            compute_histograms = computing_histograms._compute_partition_privacy_id_count_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_partition_privacy_id_count_histogram

        histogram = list(compute_histograms(input, backend))[0]
        self.assertEqual(hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION,
                         histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty',
                 input=[],
                 expected_cross_partition=[],
                 expected_per_partition=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1),
                       (1, 1)],  # (privacy_id, partition)
                expected_cross_partition=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2)
                ],
                expected_per_partition=[
                    hist.FrequencyBin(lower=1, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected_cross_partition=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ],
                expected_per_partition=[
                    hist.FrequencyBin(lower=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected_cross_partition=[
                    hist.FrequencyBin(lower=1, count=1, sum=1, max=1),
                ],
                expected_per_partition=[
                    hist.FrequencyBin(lower=100, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected_cross_partition=[
                    hist.FrequencyBin(lower=1230, count=1, sum=1234, max=1234),
                ],
                expected_per_partition=[
                    hist.FrequencyBin(lower=1, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected_cross_partition=[
                    hist.FrequencyBin(lower=15, count=2, sum=30, max=15),
                ],
                expected_per_partition=[
                    hist.FrequencyBin(lower=1, count=30, sum=30, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids',
                input=[(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 0),
                       (1, 2)],  # (privacy_id, partition)
                expected_cross_partition=[
                    hist.FrequencyBin(lower=2, count=2, sum=4, max=2),
                ],
                expected_per_partition=[
                    hist.FrequencyBin(lower=1, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, count=1, sum=2, max=2),
                    hist.FrequencyBin(lower=3, count=1, sum=3, max=3),
                ])),
        pre_aggregated=(False, True))
    def test_compute_contribution_histograms(self, testcase_name, input,
                                             expected_cross_partition,
                                             expected_per_partition,
                                             pre_aggregated):
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: 0)
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(input, backend,
                                                 data_extractors)
            data_extractors = pipeline_dp.PreAggregateExtractors(
                partition_extractor=lambda x: x[0],
                preaggregate_extractor=lambda x: x[1])
            compute_histograms = computing_histograms.compute_dataset_histograms_on_preaggregated_data
        else:
            compute_histograms = computing_histograms.compute_dataset_histograms

        histograms = list(compute_histograms(input, data_extractors, backend))
        self.assertLen(histograms, 1)
        histograms = histograms[0]

        self.assertEqual(hist.HistogramType.L0_CONTRIBUTIONS,
                         histograms.l0_contributions_histogram.name)
        self.assertEqual(hist.HistogramType.L1_CONTRIBUTIONS,
                         histograms.l1_contributions_histogram.name)
        self.assertListEqual(expected_cross_partition,
                             histograms.l0_contributions_histogram.bins)
        self.assertEqual(hist.HistogramType.LINF_CONTRIBUTIONS,
                         histograms.linf_contributions_histogram.name)
        self.assertListEqual(expected_per_partition,
                             histograms.linf_contributions_histogram.bins)


if __name__ == '__main__':
    absltest.main()
