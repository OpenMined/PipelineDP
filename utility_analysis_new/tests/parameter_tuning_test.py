from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from utility_analysis_new import parameter_tuning
from utility_analysis_new.parameter_tuning import FrequencyBin


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
        histogram = list(histogram)[0]  # the output is 1 element collection

        self.assertEqual("histogram_name", histogram.name)
        self.assertSequenceEqual(expected, histogram.bins)

    def test_list_to_contribution_histograms(self):
        histogram1 = parameter_tuning.ContributionHistogram(
            "CrossPartitionHistogram", None)
        histogram2 = parameter_tuning.ContributionHistogram(
            "PerPartitionHistogram", None)
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
        self.assertSequenceEqual(expected, histogram.bins)

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
        histogram = list(histogram)[0]
        print(histogram)
        self.assertEqual("PerPartitionHistogram", histogram.name)
        self.assertSequenceEqual(expected, histogram.bins)

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
        histograms = list(histograms)[0]
        print(histograms)

        self.assertEqual("CrossPartitionHistogram",
                         histograms.cross_partition_histogram.name)
        self.assertSequenceEqual(expected_cross_partition,
                                 histograms.cross_partition_histogram.bins)
        self.assertEqual("PerPartitionHistogram",
                         histograms.per_partition_histogram.name)
        self.assertSequenceEqual(expected_per_partition,
                                 histograms.per_partition_histogram.bins)


if __name__ == '__main__':
    absltest.main()
