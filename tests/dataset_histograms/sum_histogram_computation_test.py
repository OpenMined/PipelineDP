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
"""Tests for computing sum dataset histograms."""

from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import sum_histogram_computation
from analysis import pre_aggregation


class LowerUpperGeneratorTest(parameterized.TestCase):

    def test(self):
        g = sum_histogram_computation.LowerUpperGenerator(0, 10, num_buckets=20)
        self.assertEqual(g.bucket_len, 0.5)
        self.assertEqual(g.get_bucket_index(-1), 0)
        self.assertEqual(g.get_bucket_index(0), 0)
        self.assertEqual(g.get_bucket_index(0.5), 1)
        self.assertEqual(g.get_bucket_index(5.1), 10)
        self.assertEqual(g.get_bucket_index(10), 19)
        self.assertEqual(g.get_bucket_index(11), 19)

    @parameterized.parameters(
        (0, 0, 0), (1.223, 1.2, 1.3), (-1.223, -1.3, -1.2),
        (0.00000223, 0.0000022, 0.0000023), (134.223, 130, 140),
        (2334567, 2300000, 2400000))
    def test_log_buckets(self, x: float, expected_lower: float,
                         expected_upper: float):
        lower, upper = sum_histogram_computation._get_log_lower_upper(x)
        self.assertAlmostEqual(lower, expected_lower)
        self.assertAlmostEqual(upper, expected_upper)


class SumHistogramComputationTest(parameterized.TestCase):

    @parameterized.product(
        (
            dict(testcase_name='empty',
                 input=lambda: [],
                 expected=lambda: [],
                 expected_log=lambda: []),
            dict(
                testcase_name='small_histogram',
                input=lambda: [((1, 1), 0.5), ((1, 2), 1.5), (
                    (2, 1), -2.5), (
                        (1, 1), 0.5)],  # ((privacy_id, partition), value)
                expected=lambda: [
                    # step is (1.5 - (-2.5)) / 1e4 = 0.0004,
                    # ((2, 1), -2.5)
                    hist.FrequencyBin(lower=-2.5,
                                      upper=-2.5004,
                                      count=1,
                                      sum=-2.5,
                                      min=-2.5,
                                      max=-2.5),
                    # 2 times ((1, 1), 0.5), they are summed up and put into a
                    # bin as one.
                    hist.FrequencyBin(lower=1.0,
                                      upper=-1.0004,
                                      count=1,
                                      sum=1.0,
                                      min=1.0,
                                      max=1.0),
                    # ((1, 1, 1.5), 1.5 is max and not included,
                    # therefore 1.5 - 0.0004 = 1.4996.
                    hist.FrequencyBin(lower=1.4996,
                                      upper=1.5,
                                      count=1,
                                      sum=1.5,
                                      min=1.5,
                                      max=1.5),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=-2.6,
                                      upper=-2.5,
                                      count=1,
                                      sum=-2.5,
                                      max=-2.5,
                                      min=-2.5),
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.1,
                                      count=1,
                                      sum=1.0,
                                      max=1.0,
                                      min=1.0),
                    hist.FrequencyBin(lower=1.5,
                                      upper=1.6,
                                      count=1,
                                      sum=1.5,
                                      max=1.5,
                                      min=1.5)
                ]),
            dict(
                testcase_name='Different privacy ids, 1 equal contribution',
                input=lambda: [((i, i), 1) for i in range(100)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1, count=100, sum=100, min=1, max=1),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1.1, count=100, sum=100, min=1, max=1),
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
                                      min=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(lower=9999,
                                      upper=1000,
                                      count=2,
                                      sum=19999,
                                      min=9999,
                                      max=10000)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(
                        lower=0, upper=0, count=1, sum=0, max=0, min=0)
                ] + [
                    hist.FrequencyBin(
                        lower=i, upper=i + 0.1, count=1, sum=i, max=i, min=i)
                    for i in range(1, 10)
                ] + [
                    hist.FrequencyBin(
                        lower=i, upper=i + 1, count=1, sum=i, max=i, min=i)
                    for i in range(10, 100)
                ] + [
                    hist.FrequencyBin(lower=i,
                                      upper=i + 10,
                                      count=10,
                                      sum=(i + i + 9) * 10 // 2,
                                      max=i + 9,
                                      min=i) for i in range(100, 1000, 10)
                ] + [
                    hist.FrequencyBin(lower=i,
                                      upper=i + 100,
                                      count=100,
                                      sum=(i + i + 99) * 100 // 2,
                                      max=i + 99,
                                      min=i) for i in range(1000, 10000, 100)
                ] + [
                    hist.FrequencyBin(lower=10000,
                                      upper=11000,
                                      count=1,
                                      sum=10000,
                                      max=10000,
                                      min=10000)
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 '
                'partition',
                input=lambda: [(
                    (0, 0), 1.0)] * 100,  # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(lower=100.0,
                                      upper=100.0,
                                      count=1,
                                      sum=100.0,
                                      min=100.0,
                                      max=100.0),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=100.0,
                                      upper=110.0,
                                      count=1,
                                      sum=100.0,
                                      min=100.0,
                                      max=100.0)
                ]),
            dict(
                testcase_name=
                '1 privacy id many equal contributions to many partitions',
                input=lambda: [((0, i), 1.0) for i in range(1234)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.0,
                                      count=1234,
                                      sum=1234.0,
                                      min=1.0,
                                      max=1),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.1,
                                      count=1234,
                                      sum=1234.0,
                                      min=1.0,
                                      max=1.0)
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
                                      min=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(lower=9999,
                                      upper=1000,
                                      count=2,
                                      sum=19999,
                                      min=9999,
                                      max=10000)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(
                        lower=0, upper=0, count=1, sum=0, max=0, min=0)
                ] + [
                    hist.FrequencyBin(
                        lower=i, upper=i + 0.1, count=1, sum=i, max=i, min=i)
                    for i in range(1, 10)
                ] + [
                    hist.FrequencyBin(
                        lower=i, upper=i + 1, count=1, sum=i, max=i, min=i)
                    for i in range(10, 100)
                ] + [
                    hist.FrequencyBin(lower=i,
                                      upper=i + 10,
                                      count=10,
                                      sum=(i + i + 9) * 10 // 2,
                                      max=i + 9,
                                      min=i) for i in range(100, 1000, 10)
                ] + [
                    hist.FrequencyBin(lower=i,
                                      upper=i + 100,
                                      count=100,
                                      sum=(i + i + 99) * 100 // 2,
                                      max=i + 99,
                                      min=i) for i in range(1000, 10000, 100)
                ] + [
                    hist.FrequencyBin(lower=10000,
                                      upper=11000,
                                      count=1,
                                      sum=10000,
                                      max=10000,
                                      min=10000)
                ]),
            dict(
                testcase_name=
                '2 privacy ids, same partitions equally contributed',
                input=lambda: [((0, i), 1.0) for i in range(15)] + [(
                    (1, i), 1.0) for i in range(10, 25)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.0, count=30, sum=30, min=1, max=1),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(
                        lower=1.0, upper=1.1, count=30, sum=30, min=1, max=1),
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
                                      upper=-1.0002,
                                      count=15,
                                      sum=-15,
                                      min=-1,
                                      max=-1),
                    hist.FrequencyBin(
                        lower=0.9998,
                        upper=1, count=15, sum=15, min=1, max=1),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=-1.1,
                                      upper=-1.0,
                                      count=15,
                                      sum=-15.0,
                                      max=-1.0,
                                      min=-1.0),
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.1,
                                      count=15,
                                      sum=15.0,
                                      max=1.0,
                                      min=1.0)
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_linf_sum_contributions_histogram(self, testcase_name,
                                                      input, expected,
                                                      expected_log,
                                                      pre_aggregated):
        # Lambdas are used for returning input and expected. Passing lists
        # instead leads to printing whole lists as test names in the output.
        # That complicates debugging.
        input = input()
        expected = expected()
        expected_log = expected_log()
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
            compute_histograms = sum_histogram_computation._compute_linf_sum_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = sum_histogram_computation._compute_linf_sum_contributions_histogram
        histogram, log_histogram = list(compute_histograms(input, backend))
        if not input:
            self.assertEqual(histogram, [])
            self.assertEqual(log_histogram, [])
        else:
            self.assertEqual(hist.HistogramType.LINF_SUM_CONTRIBUTIONS,
                             histogram.name)
            self.assertListEqual(expected, histogram.bins)
            self.assertEqual(hist.HistogramType.LINF_SUM_CONTRIBUTIONS_LOG,
                             log_histogram.name)
            self.assertListEqual(expected_log, log_histogram.bins)

    @parameterized.parameters(False, True)
    def test_compute_linf_sum_contributions_histogram_2_columns(
            self, pre_aggregated: bool):
        # format: ((privacy_id, partition), value: tuple)
        data = [((0, 0), (1, 10)), ((0, 1), (2, 20)), ((0, 1), (3, 30)),
                ((1, 0), (5, 50))]
        backend = pipeline_dp.LocalBackend()
        expected = [
            hist.Histogram(hist.HistogramType.LINF_SUM_CONTRIBUTIONS, [
                hist.FrequencyBin(
                    lower=1.0, upper=1.0004, count=1, sum=1, min=1, max=1),
                hist.FrequencyBin(
                    lower=4.9996, upper=5.0, count=2, sum=10, min=5, max=5)
            ]),
            hist.Histogram(hist.HistogramType.LINF_SUM_CONTRIBUTIONS, [
                hist.FrequencyBin(
                    lower=10.0, upper=10.004, count=1, sum=10, min=10, max=10),
                hist.FrequencyBin(
                    lower=49.996, upper=50.0, count=2, sum=100, min=50, max=50)
            ])
        ]
        expected_log = [
            hist.Histogram(hist.HistogramType.LINF_SUM_CONTRIBUTIONS_LOG, [
                hist.FrequencyBin(
                    lower=1.0, upper=1.1, count=1, sum=1, min=1, max=1),
                hist.FrequencyBin(
                    lower=5.0, upper=5.1, count=2, sum=10, min=5, max=5)
            ]),
            hist.Histogram(hist.HistogramType.LINF_SUM_CONTRIBUTIONS_LOG, [
                hist.FrequencyBin(
                    lower=10.0, upper=11, count=1, sum=10, min=10, max=10),
                hist.FrequencyBin(
                    lower=50, upper=51, count=2, sum=100, min=50, max=50)
            ])
        ]
        if pre_aggregated:
            data = list(
                pre_aggregation.preaggregate(
                    data,
                    backend,
                    data_extractors=pipeline_dp.DataExtractors(
                        privacy_id_extractor=lambda x: x[0][0],
                        partition_extractor=lambda x: x[0][1],
                        value_extractor=lambda x: x[1])))

            compute_histograms = sum_histogram_computation._compute_linf_sum_contributions_histogram_on_preaggregated_data
        else:
            compute_histograms = sum_histogram_computation._compute_linf_sum_contributions_histogram
        histogram, log_histogram = list(compute_histograms(data, backend))
        self.assertListEqual(histogram, expected)
        self.assertListEqual(log_histogram, expected_log)

    @parameterized.product(
        (
            dict(testcase_name='empty histogram',
                 input=lambda: [],
                 expected=lambda: [],
                 expected_log=lambda: []),
            dict(
                testcase_name='small_histogram',
                input=lambda: [((1, 1), 0.5), ((1, 2), 1.5), (
                    (2, 1), -2.5), (
                        (1, 1), 0.5)],  # ((privacy_id, partition), value)
                expected=lambda: [
                    # Bucket step = 3/10**4 = 0.0003
                    hist.FrequencyBin(lower=-1.5,
                                      upper=-1.4997,
                                      count=1,
                                      sum=-1.5,
                                      min=-1.5,
                                      max=-1.5),
                    hist.FrequencyBin(lower=1.4996999999999998,
                                      upper=1.5,
                                      count=1,
                                      sum=1.5,
                                      min=1.5,
                                      max=1.5)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=-1.6,
                                      upper=-1.5,
                                      count=1,
                                      sum=-1.5,
                                      min=-1.5,
                                      max=-1.5),
                    hist.FrequencyBin(lower=1.5,
                                      upper=1.6,
                                      count=1,
                                      sum=1.5,
                                      min=1.5,
                                      max=1.5)
                ]),
            dict(
                testcase_name='Different privacy ids, 1 equal contribution and '
                'different partition keys',
                input=lambda: [((i, i), 1) for i in range(100)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1, count=100, sum=100, min=1, max=1),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(
                        lower=1, upper=1.1, count=100, sum=100, min=1, max=1)
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
                                      min=i,
                                      max=i) for i in range(9999)
                ] + [
                    hist.FrequencyBin(lower=9999,
                                      upper=1000,
                                      count=2,
                                      sum=19999,
                                      min=9999,
                                      max=10000)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(
                        lower=0, upper=0, count=1, sum=0, max=0, min=0)
                ] + [
                    hist.FrequencyBin(
                        lower=i, upper=i + 0.1, count=1, sum=i, max=i, min=i)
                    for i in range(1, 10)
                ] + [
                    hist.FrequencyBin(
                        lower=i, upper=i + 1, count=1, sum=i, max=i, min=i)
                    for i in range(10, 100)
                ] + [
                    hist.FrequencyBin(lower=i,
                                      upper=i + 10,
                                      count=10,
                                      sum=(i + i + 9) * 10 // 2,
                                      max=i + 9,
                                      min=i) for i in range(100, 1000, 10)
                ] + [
                    hist.FrequencyBin(lower=i,
                                      upper=i + 100,
                                      count=100,
                                      sum=(i + i + 99) * 100 // 2,
                                      max=i + 99,
                                      min=i) for i in range(1000, 10000, 100)
                ] + [
                    hist.FrequencyBin(lower=10000,
                                      upper=11000,
                                      count=1,
                                      sum=10000,
                                      max=10000,
                                      min=10000)
                ]),
            dict(
                testcase_name='1 privacy id many contributions to 1 '
                'partition',
                input=lambda: [(
                    (0, 0), 1.0)] * 100,  # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(lower=100.0,
                                      upper=100.0,
                                      count=1,
                                      sum=100.0,
                                      min=100.0,
                                      max=100.0)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=100.0,
                                      upper=110.0,
                                      count=1,
                                      sum=100.0,
                                      min=100.0,
                                      max=100.0)
                ]),
            dict(
                testcase_name=
                '1 privacy id many equal contributions to many partitions',
                input=lambda: [((0, i), 1.0) for i in range(1234)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.0,
                                      count=1234,
                                      sum=1234.0,
                                      min=1,
                                      max=1),
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.1,
                                      count=1234,
                                      sum=1234.0,
                                      min=1,
                                      max=1),
                ]),
            dict(
                testcase_name=
                '2 privacy ids, same partitions equally contributed',
                input=lambda: [((0, i), 1.0) for i in range(15)] + [(
                    (1, i), 1.0) for i in range(10, 25)],
                # ((privacy_id, partition), value)
                expected=lambda: [
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.0001,
                                      count=20,
                                      sum=20.0,
                                      min=1.0,
                                      max=1.0),
                    hist.FrequencyBin(lower=1.9999,
                                      upper=2.0,
                                      count=5,
                                      sum=10.0,
                                      min=2.0,
                                      max=2.0)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.1,
                                      count=20,
                                      sum=20.0,
                                      min=1.0,
                                      max=1.0),
                    hist.FrequencyBin(lower=2.0,
                                      upper=2.1,
                                      count=5,
                                      sum=10.0,
                                      min=2.0,
                                      max=2.0)
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
                                      min=-1.0,
                                      max=-1.0),
                    hist.FrequencyBin(lower=0.0,
                                      upper=0.00019999999999997797,
                                      count=5,
                                      sum=0.0,
                                      min=0,
                                      max=0.0),
                    hist.FrequencyBin(lower=0.9998,
                                      upper=1.0,
                                      count=10,
                                      sum=10.0,
                                      min=1.0,
                                      max=1.0)
                ],
                expected_log=lambda: [
                    hist.FrequencyBin(lower=-1.1,
                                      upper=-1.0,
                                      count=10,
                                      sum=-10.0,
                                      min=-1.0,
                                      max=-1.0),
                    hist.FrequencyBin(
                        lower=0.0, upper=0.0, count=5, sum=0.0, min=0, max=0.0),
                    hist.FrequencyBin(lower=1.0,
                                      upper=1.1,
                                      count=10,
                                      sum=10.0,
                                      min=1.0,
                                      max=1.0)
                ]),
        ),
        pre_aggregated=(False, True))
    def test_compute_partition_sum_histogram(self, testcase_name, input,
                                             expected, expected_log,
                                             pre_aggregated):
        input = input()
        expected = expected()
        expected_log = expected_log()
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
            compute_histograms = sum_histogram_computation._compute_partition_sum_histogram_on_preaggregated_data
        else:
            compute_histograms = sum_histogram_computation._compute_partition_sum_histogram
        histogram, log_histogram = list(compute_histograms(input, backend))
        if not input:
            self.assertEqual(histogram, [])
        else:
            self.assertEqual(hist.HistogramType.SUM_PER_PARTITION,
                             histogram.name)
            self.assertListEqual(expected, histogram.bins)
            self.assertEqual(hist.HistogramType.SUM_PER_PARTITION_LOG,
                             log_histogram.name)
            self.assertListEqual(expected_log, log_histogram.bins)

    @parameterized.parameters(False, True)
    def test_compute_partition_sum_histogram_2_columns(self,
                                                       pre_aggregated: bool):
        # format: ((privacy_id, partition), value: tuple)
        data = [((0, 0), (1, 10)), ((0, 1), (2, 20)), ((0, 1), (3, 30)),
                ((1, 0), (5, 50))]
        backend = pipeline_dp.LocalBackend()
        expected = [
            hist.Histogram(hist.HistogramType.SUM_PER_PARTITION, [
                hist.FrequencyBin(
                    lower=5.0, upper=5.0001, count=1, sum=5, min=5, max=5),
                hist.FrequencyBin(
                    lower=5.9999, upper=6.0, count=1, sum=6, min=6, max=6)
            ]),
            hist.Histogram(hist.HistogramType.SUM_PER_PARTITION, [
                hist.FrequencyBin(
                    lower=50.0, upper=50.001, count=1, sum=50, min=50, max=50),
                hist.FrequencyBin(
                    lower=59.999, upper=60.0, count=1, sum=60, min=60, max=60)
            ])
        ]
        expected_log = [
            hist.Histogram(hist.HistogramType.SUM_PER_PARTITION_LOG, [
                hist.FrequencyBin(
                    lower=5.0, upper=5.1, count=1, sum=5, min=5, max=5),
                hist.FrequencyBin(
                    lower=6.0, upper=6.1, count=1, sum=6, min=6, max=6)
            ]),
            hist.Histogram(hist.HistogramType.SUM_PER_PARTITION_LOG, [
                hist.FrequencyBin(
                    lower=50.0, upper=51, count=1, sum=50, min=50, max=50),
                hist.FrequencyBin(
                    lower=60.0, upper=61, count=1, sum=60, min=60, max=60)
            ])
        ]
        if pre_aggregated:
            data = list(
                pre_aggregation.preaggregate(
                    data,
                    backend,
                    data_extractors=pipeline_dp.DataExtractors(
                        privacy_id_extractor=lambda x: x[0][0],
                        partition_extractor=lambda x: x[0][1],
                        value_extractor=lambda x: x[1])))

            compute_histograms = sum_histogram_computation._compute_partition_sum_histogram_on_preaggregated_data
        else:
            compute_histograms = sum_histogram_computation._compute_partition_sum_histogram
        histogram, log_histogram = list(compute_histograms(data, backend))
        self.assertListEqual(histogram, expected)
        self.assertListEqual(log_histogram, expected_log)


if __name__ == '__main__':
    absltest.main()
