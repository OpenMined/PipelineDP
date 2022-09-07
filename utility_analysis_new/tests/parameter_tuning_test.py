from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from utility_analysis_new import parameter_tuning
from utility_analysis_new.parameter_tuning import FrequencyBin


class ParameterTuning(parameterized.TestCase):

    def testToBinLower(self):
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
    def testComputeFrequencyHistogram(self, input, expected):
        backend = pipeline_dp.LocalBackend()
        histogram = parameter_tuning._compute_frequency_histogram(
            input, backend)
        histogram = list(histogram)[0]  # the output is 1 element collection

        self.assertSequenceEqual(histogram, expected)


if __name__ == '__main__':
    absltest.main()
