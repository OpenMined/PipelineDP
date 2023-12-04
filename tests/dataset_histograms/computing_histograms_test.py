# Copyright 2023 OpenMined.
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
"""Tests for computing dataset histograms."""

from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import computing_histograms
from analysis import pre_aggregation


class ComputingHistogramsTest(parameterized.TestCase):

    def test_to_bin_lower_upper_logarithmic(self):
        to_bin_lower = computing_histograms._to_bin_lower_upper_logarithmic
        self.assertEqual(to_bin_lower(1), (1, 2))
        self.assertEqual(to_bin_lower(999), (999, 1000))
        self.assertEqual(to_bin_lower(1000), (1000, 1010))
        self.assertEqual(to_bin_lower(1001), (1000, 1010))
        self.assertEqual(to_bin_lower(1012), (1010, 1020))
        self.assertEqual(to_bin_lower(2022), (2020, 2030))
        self.assertEqual(to_bin_lower(12522), (12500, 12600))
        self.assertEqual(to_bin_lower(10**9 + 10**7 + 1234),
                         (10**9 + 10**7, 10**9 + 2 * 10**7))

    def test_bin_lower_index(self):
        lowers = [0.5, 1.2, 3.6, 5]
        to_bin_lower_idx = computing_histograms._bin_lower_index
        with self.assertRaises(AssertionError):
            to_bin_lower_idx(lowers, 0.3)
        self.assertEqual(to_bin_lower_idx(lowers, 0.5), 0)
        self.assertEqual(to_bin_lower_idx(lowers, 1), 0)
        self.assertEqual(to_bin_lower_idx(lowers, 1.2), 1)
        self.assertEqual(to_bin_lower_idx(lowers, 1.3), 1)
        self.assertEqual(to_bin_lower_idx(lowers, 3.6), 2)
        self.assertEqual(to_bin_lower_idx(lowers, 5), 2)
        with self.assertRaises(AssertionError):
            to_bin_lower_idx(lowers, 5.1)

    @parameterized.named_parameters(
        dict(testcase_name='empty', input=[], expected=[]),
        dict(testcase_name='small_histogram',
             input=[3, 3, 1, 1, 2, 10],
             expected=[
                 hist.FrequencyBin(lower=1, upper=2, count=2, sum=2, max=1),
                 hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2),
                 hist.FrequencyBin(lower=3, upper=4, count=2, sum=6, max=3),
                 hist.FrequencyBin(lower=10, upper=11, count=1, sum=10, max=10)
             ]),
        dict(testcase_name='histogram_with_bins_wider_1',
             input=[1005, 3, 12345, 12346],
             expected=[
                 hist.FrequencyBin(lower=3, upper=4, count=1, sum=3, max=3),
                 hist.FrequencyBin(lower=1000,
                                   upper=1010,
                                   count=1,
                                   sum=1005,
                                   max=1005),
                 hist.FrequencyBin(lower=12300,
                                   upper=12400,
                                   count=2,
                                   sum=24691,
                                   max=12346)
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
        histogram1 = hist.Histogram(hist.HistogramType.L0_CONTRIBUTIONS, [])
        histogram2 = hist.Histogram(hist.HistogramType.L1_CONTRIBUTIONS, [])
        histogram3 = hist.Histogram(hist.HistogramType.LINF_CONTRIBUTIONS, [])
        histogram4 = hist.Histogram(hist.HistogramType.LINF_SUM_CONTRIBUTIONS,
                                    [])
        histogram5 = hist.Histogram(hist.HistogramType.COUNT_PER_PARTITION, [])
        histogram6 = hist.Histogram(
            hist.HistogramType.COUNT_PRIVACY_ID_PER_PARTITION, [])
        histograms = computing_histograms._list_to_contribution_histograms([
            histogram2, histogram1, histogram3, histogram4, histogram6,
            histogram5
        ])
        self.assertEqual(histogram1, histograms.l0_contributions_histogram)
        self.assertEqual(histogram2, histograms.l1_contributions_histogram)
        self.assertEqual(histogram3, histograms.linf_contributions_histogram)
        self.assertEqual(histogram4,
                         histograms.linf_sum_contributions_histogram)
        self.assertEqual(histogram5, histograms.count_per_partition_histogram)
        self.assertEqual(histogram6, histograms.count_privacy_id_per_partition)

    @parameterized.product(
        (
            dict(testcase_name='empty', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1230, upper=1240, count=1, sum=1234, max=1234),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, max=15),
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
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i // 2) for i in range(1235)
                      ],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1230, upper=1240, count=1, sum=1235, max=1235),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, max=15),
                ]),
            dict(
                testcase_name='3 privacy ids',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)] +
                [(2, i) for i in range(11)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=11, upper=12, count=1, sum=11, max=11),
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, max=15),
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
                    hist.FrequencyBin(lower=1, upper=2, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=30, sum=30,
                                      max=1),
                ]),
            dict(
                testcase_name='2 privacy ids',
                input=[(0, 0), (0, 0), (0, 1), (1, 0), (1, 0), (1, 0),
                       (1, 2)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2),
                    hist.FrequencyBin(lower=3, upper=4, count=1, sum=3, max=3),
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
            dict(testcase_name='empty', input=lambda: [], expected=lambda: []),
            dict(
                testcase_name='small_histogram',
                input=lambda: [((1, 1), 0.5), ((1, 2), 1.5), (
                    (2, 1), -2.5), (
                        (1, 1), 0.5)],  # ((privacy_id, partition), value)
                expected=lambda: [
                    # step is (1.5 - (-2.5)) / 1e4 = 0.0004,
                    # ((2, 1), -2.5)
                    hist.FrequencyBin(
                        lower=-2.5, upper=-2.5004, count=1, sum=-2.5, max=-2.5),
                    # 2 times ((1, 1), 0.5), they are summed up and put into a
                    # bin as one.
                    hist.FrequencyBin(
                        lower=1.0, upper=-1.0004, count=1, sum=1.0, max=1.0),
                    # ((1, 1, 1.5), 1.5 is max and not included,
                    # therefore 1.5 - 0.0004 = 1.4996.
                    hist.FrequencyBin(
                        lower=1.4996,
                        upper=1.5, count=1, sum=1.5, max=1.5),
                ]),
            dict(
                testcase_name='Different privacy ids, 1 equal contribution',
                input=lambda: [((i, i), 1) for i in range(100)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='Different privacy ids, 1 different contribution',
                input=lambda: [((i, i), i) for i in range(10001)],
                # ((privacy_id, partition), value)
                # step is 1e4 / 1e4 = 1, therefore 1e4 - 1 = 9999.
                expected=lambda: [
                    hist.FrequencyBin(lower=float(i),
                                      upper=float(i + 1),
                                      count=1,
                                      sum=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(
                        lower=9999, upper=1000, count=2, sum=19999, max=10000)
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 '
                'partition',
                input=lambda: [(
                    (0, 0), 1.0)] * 100,  # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=100.0, upper=100.0, count=1, sum=100.0, max=100.0
                    ),
                ]),
            dict(
                testcase_name=
                '1 privacy id many equal contributions to many partition',
                input=lambda: [((0, i), 1.0) for i in range(1234)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0, count=1234, sum=1234.0, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many different contributions to many partition',
                input=lambda: [((0, i), i) for i in range(10001)],
                # ((privacy_id, partition), value)
                # step is 1e4 / 1e4 = 1, therefore 1e4 - 1 = 9999.
                expected=lambda: [
                    hist.FrequencyBin(lower=float(i),
                                      upper=float(i + 1),
                                      count=1,
                                      sum=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(
                        lower=9999, upper=1000, count=2, sum=19999, max=10000)
                ]),
            dict(
                testcase_name=
                '2 privacy ids, same partitions equally contributed',
                input=lambda: [((0, i), 1.0) for i in range(15)] + [(
                    (1, i), 1.0) for i in range(10, 25)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0, count=30, sum=30, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions differently '
                'contributed',
                input=lambda: [((0, i), -1.0) for i in range(15)] + [(
                    (1, i), 1.0) for i in range(10, 25)],
                # ((privacy_id, partition), value)
                # step = (1 - (-1)) / 1e4 = 0.0002,
                # therefore last lower is 1 - 0.0002 = 0.9998.
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=-1.0, upper=-1.0002, count=15, sum=-15, max=-1),
                    hist.
                    FrequencyBin(lower=0.9998, upper=1, count=15, sum=15, max=1
                                ),
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_linf_sum_contributions_histogram(self, testcase_name,
                                                      input, expected,
                                                      pre_aggregated):
        # Lambdas are used for returning input and expected. Passing lists
        # instead lead to printing whole lists as test names in the output.
        # That complicates debugging.
        input = input()
        expected = expected()
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = pre_aggregation.preaggregate(
                input,
                backend,
                data_extractors=pipeline_dp.DataExtractors(
                    privacy_id_extractor=lambda x: x[0][0],
                    partition_extractor=lambda x: x[0][1],
                    value_extractor=lambda x: x[1]))
            compute_histograms = computing_histograms._compute_linf_sum_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_linf_sum_contributions_histogram
        histogram = list(compute_histograms(input, backend))
        self.assertLen(histogram, 1)
        histogram = histogram[0]
        self.assertEqual(hist.HistogramType.LINF_SUM_CONTRIBUTIONS,
                         histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(testcase_name='empty histogram', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1),
                       (1, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=3, upper=3, count=1, sum=3, max=3)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)] * 100,  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, max=100),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partitions',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=20, sum=20,
                                      max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=5, sum=10, max=2),
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
            dict(testcase_name='empty histogram', input=[], expected=[]),
            dict(
                testcase_name='small_histogram',
                input=[(1, 1), (1, 2), (2, 1)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2)
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=[(i, i) for i in range(100)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=[(0, 0)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partitions',
                input=[(0, i) for i in range(1234)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, max=1),
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=[(0, i) for i in range(15)] +
                [(1, i) for i in range(10, 25)],  # (privacy_id, partition)
                expected=[
                    hist.FrequencyBin(lower=1, upper=2, count=20, sum=20,
                                      max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=5, sum=10, max=2),
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
            dict(testcase_name='empty histogram',
                 input=lambda: [],
                 expected=lambda: []),
            dict(
                testcase_name='small_histogram',
                input=lambda: [((1, 1), 0.5), ((1, 2), 1.5), (
                    (2, 1), -2.5), (
                        (1, 1), 0.5)],  # ((privacy_id, partition), value)
                expected=lambda: [
                    # Bucket step = 3/10**4 = 0.0003
                    hist.FrequencyBin(
                        lower=-1.5, upper=-1.4997, count=1, sum=-1.5, max=-1.5),
                    hist.FrequencyBin(lower=1.4996999999999998,
                                      upper=1.5,
                                      count=1,
                                      sum=1.5,
                                      max=1.5)
                ]),
            dict(
                testcase_name='Different privacy ids, 1 equal contribution and '
                'different partition keys',
                input=lambda: [((i, i), 1) for i in range(100)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1, count=100, sum=100, max=1),
                ]),
            dict(
                testcase_name='Different privacy ids, 1 different contribution',
                input=lambda: [((i, i), i) for i in range(10001)],
                # ((privacy_id, partition), value)
                # step is 1e4 / 1e4 = 1, therefore 1e4 - 1 = 9999.
                expected=lambda: [
                    hist.FrequencyBin(lower=float(i),
                                      upper=float(i + 1),
                                      count=1,
                                      sum=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(
                        lower=9999, upper=1000, count=2, sum=19999, max=10000)
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 '
                'partition',
                input=lambda: [(
                    (0, 0), 1.0)] * 100,  # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=100.0, upper=100.0, count=1, sum=100.0, max=100.0
                    ),
                ]),
            dict(
                testcase_name=
                '1 privacy id many equal contributions to many partitions',
                input=lambda: [((0, i), 1.0) for i in range(1234)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0, count=1234, sum=1234.0, max=1),
                ]),
            dict(
                testcase_name=
                '1 privacy id many different contributions to many partitions',
                input=lambda: [((0, i), i) for i in range(10001)],
                # ((privacy_id, partition), value)
                # step is 1e4 / 1e4 = 1, therefore 1e4 - 1 = 9999.
                expected=lambda: [
                    hist.FrequencyBin(lower=float(i),
                                      upper=float(i + 1),
                                      count=1,
                                      sum=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(
                        lower=9999, upper=1000, count=2, sum=19999, max=10000)
                ]),
            dict(
                testcase_name=
                '2 privacy ids, same partitions equally contributed',
                input=lambda: [((0, i), 1.0) for i in range(15)] + [(
                    (1, i), 1.0) for i in range(10, 25)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0001, count=20, sum=20.0, max=1.0),
                    hist.FrequencyBin(
                        lower=1.9999,
                        upper=2.0, count=5, sum=10.0, max=2.0)
                ]),
            dict(
                testcase_name='2 privacy ids, same partitions differently '
                'contributed',
                input=lambda: [((0, i), -1.0) for i in range(15)] + [(
                    (1, i), 1.0) for i in range(10, 25)],
                # ((privacy_id, partition), value)
                # step = (1 - (-1)) / 1e4 = 0.0002,
                # therefore last lower is 1 - 0.0002 = 0.9998.
                expected=lambda: [
                    hist.FrequencyBin(lower=-1.0,
                                      upper=-0.9998,
                                      count=10,
                                      sum=-10.0,
                                      max=-1.0),
                    hist.FrequencyBin(lower=0.0,
                                      upper=0.00019999999999997797,
                                      count=5,
                                      sum=0.0,
                                      max=0.0),
                    hist.FrequencyBin(
                        lower=0.9998,
                        upper=1.0, count=10, sum=10.0, max=1.0)
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_partition_sum_histogram(self, testcase_name, input,
                                             expected, pre_aggregated):
        input = input()
        expected = expected()
        backend = pipeline_dp.LocalBackend()
        if pre_aggregated:
            input = list(
                pre_aggregation.preaggregate(
                    input,
                    backend,
                    data_extractors=pipeline_dp.DataExtractors(
                        privacy_id_extractor=lambda x: x[0][0],
                        partition_extractor=lambda x: x[0][1],
                        value_extractor=lambda x: x[1])))
            compute_histograms = computing_histograms._compute_partition_sum_histogram_on_preaggregated_data
        else:
            compute_histograms = computing_histograms._compute_partition_sum_histogram
        histogram = list(compute_histograms(input, backend))
        self.assertLen(histogram, 1)
        histogram = histogram[0]
        self.assertEqual(hist.HistogramType.SUM_PER_PARTITION, histogram.name)
        self.assertListEqual(expected, histogram.bins)

    @parameterized.product(
        (
            dict(
                testcase_name='empty',
                input=lambda: [],
                expected_cross_partition=lambda: [],
                expected_per_partition=lambda: [],
                expected_sum_per_partition=lambda: [],
            ),
            dict(
                testcase_name='small_histogram',
                input=lambda: [(1, 1, 0.5), (1, 2, 1.5), (2, 1, -2.5),
                               (1, 1, 0.5)],  # (privacy_id, partition, value)
                expected_cross_partition=lambda: [
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2)
                ],
                expected_per_partition=lambda: [
                    hist.FrequencyBin(lower=1, upper=2, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2)
                ],
                expected_sum_per_partition=lambda: [
                    # see for explanation why these values
                    # test_compute_linf_sum_contributions_histogram.
                    hist.FrequencyBin(
                        lower=-2.5, upper=-2.5004, count=1, sum=-2.5, max=-2.5),
                    hist.FrequencyBin(
                        lower=1.0, upper=-1.0004, count=1, sum=1.0, max=1.0),
                    hist.FrequencyBin(
                        lower=1.4996,
                        upper=1.5, count=1, sum=1.5, max=1.5),
                ]),
            dict(
                testcase_name='Each privacy id, 1 contribution',
                input=lambda: [(i, i, 1.0) for i in range(100)
                              ],  # (privacy_id, partition, value)
                expected_cross_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ],
                expected_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=2, count=100, sum=100, max=1),
                ],
                expected_sum_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1, count=100, sum=100, max=1),
                ],
            ),
            dict(
                testcase_name='1 privacy id many contributions to 1 partition',
                input=lambda: [(0, 0, 1.0)] *
                100,  # (privacy_id, partition, value)
                expected_cross_partition=lambda: [
                    hist.FrequencyBin(lower=1, upper=2, count=1, sum=1, max=1),
                ],
                expected_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=100, upper=101, count=1, sum=100, max=100),
                ],
                expected_sum_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=100.0, upper=100.0, count=1, sum=100.0, max=100.0
                    ),
                ],
            ),
            dict(
                testcase_name=
                '1 privacy id many contributions to many partition',
                input=lambda: [(0, i, 1.0) for i in range(1234)
                              ],  # (privacy_id, partition, value)
                expected_cross_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1230, upper=1240, count=1, sum=1234, max=1234),
                ],
                expected_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=2, count=1234, sum=1234, max=1),
                ],
                expected_sum_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0, count=1234, sum=1234.0, max=1),
                ],
            ),
            dict(
                testcase_name='2 privacy ids, same partitions contributed',
                input=lambda: [(0, i, 1.0) for i in range(15)] + [
                    (1, i, 1.0) for i in range(10, 25)
                ],  # (privacy_id, partition, value)
                expected_cross_partition=lambda: [
                    hist.FrequencyBin(
                        lower=15, upper=16, count=2, sum=30, max=15),
                ],
                expected_per_partition=lambda: [
                    hist.FrequencyBin(lower=1, upper=2, count=30, sum=30, max=1
                                     ),
                ],
                expected_sum_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0, count=30, sum=30, max=1),
                ],
            ),
            dict(
                testcase_name='2 privacy ids',
                input=lambda: [(0, 0, 1.0), (0, 0, 1.0), (0, 1, 2.0),
                               (1, 0, 0.0), (1, 0, 1.3), (1, 0, 0.7),
                               (1, 2, 2.0)],  # (privacy_id, partition, value)
                expected_cross_partition=lambda: [
                    hist.FrequencyBin(lower=2, upper=3, count=2, sum=4, max=2),
                ],
                expected_per_partition=lambda: [
                    hist.FrequencyBin(lower=1, upper=2, count=2, sum=2, max=1),
                    hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2),
                    hist.FrequencyBin(lower=3, upper=4, count=1, sum=3, max=3),
                ],
                expected_sum_per_partition=lambda: [
                    hist.FrequencyBin(
                        lower=2.0, upper=2.0, count=4, sum=8, max=2),
                ],
            )),
        pre_aggregated=(False, True))
    def test_compute_contribution_histograms(self, testcase_name, input,
                                             expected_cross_partition,
                                             expected_per_partition,
                                             expected_sum_per_partition,
                                             pre_aggregated):
        input = input()
        expected_cross_partition = expected_cross_partition()
        expected_per_partition = expected_per_partition()
        expected_sum_per_partition = expected_sum_per_partition()
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: x[2])
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
        self.assertEqual(hist.HistogramType.LINF_SUM_CONTRIBUTIONS,
                         histograms.linf_sum_contributions_histogram.name)
        self.assertListEqual(expected_sum_per_partition,
                             histograms.linf_sum_contributions_histogram.bins)


if __name__ == '__main__':
    absltest.main()
