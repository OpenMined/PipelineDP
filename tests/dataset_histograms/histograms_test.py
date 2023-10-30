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
from typing import Union, List

from absl.testing import absltest
from absl.testing import parameterized

from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms.histograms import HistogramType, \
    FrequencyBin


def frequency_bin(lower: Union[int, float],
                  upper: Union[int, float]) -> FrequencyBin:
    return FrequencyBin(lower, upper, count=0, sum=0, max=0)


class HistogramTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='1 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1000,
                                   upper=1010,
                                   count=10,
                                   sum=10100,
                                   max=1009),
             ],
             q=[0.05, 0.1, 0.5, 0.8, 0.9],
             expected_quantiles=[1000, 1000, 1000, 1000, 1000]),
        dict(testcase_name='6 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1, upper=2, count=2, sum=2, max=1),
                 hist.FrequencyBin(lower=2, upper=3, count=1, sum=2, max=2),
                 hist.FrequencyBin(lower=3, upper=4, count=1, sum=3, max=3),
                 hist.FrequencyBin(lower=4, upper=5, count=2, sum=8, max=4),
                 hist.FrequencyBin(lower=5, upper=6, count=2, sum=10, max=5),
                 hist.FrequencyBin(lower=6, upper=7, count=1, sum=6, max=6),
                 hist.FrequencyBin(lower=10, upper=12, count=1, sum=11, max=11)
             ],
             q=[0.001, 0.05, 0.1, 0.5, 0.8, 0.9],
             expected_quantiles=[1, 1, 1, 4, 6, 10]))
    def test_quantile_contributions(self, bins, q, expected_quantiles):
        histogram = hist.Histogram("name", bins)
        output = histogram.quantiles(q)
        self.assertListEqual(expected_quantiles, output)

    @parameterized.named_parameters(
        dict(testcase_name='empty', bins=[], expected_ratios=[]),
        dict(testcase_name='1 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1000,
                                   upper=1021,
                                   count=10,
                                   sum=10100,
                                   max=1020),
             ],
             expected_ratios=[(0, 1), (1000, 100 / 10100), (1020, 0.0)]),
        dict(testcase_name='7 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1, upper=2, count=8, sum=8, max=1),
                 hist.FrequencyBin(lower=2, upper=3, count=2, sum=4, max=2),
                 hist.FrequencyBin(lower=3, upper=4, count=1, sum=3, max=3),
                 hist.FrequencyBin(lower=4, upper=5, count=2, sum=8, max=4),
                 hist.FrequencyBin(lower=5, upper=6, count=2, sum=10, max=5),
                 hist.FrequencyBin(lower=6, upper=7, count=1, sum=6, max=6),
                 hist.FrequencyBin(lower=11, upper=12, count=1, sum=11, max=11),
             ],
             expected_ratios=[(0, 1), (1, 0.66), (2, 0.48), (3, 0.34),
                              (4, 0.22), (5, 0.14), (6, 0.1), (11, 0.0)]))
    def test_ratio_dropped(self, bins, expected_ratios):
        histogram = hist.Histogram("name", bins)
        output = hist.compute_ratio_dropped(histogram)
        self.assertListEqual(output, expected_ratios)

    @parameterized.parameters(
        (HistogramType.L0_CONTRIBUTIONS, True),
        (HistogramType.L1_CONTRIBUTIONS, True),
        (HistogramType.LINF_CONTRIBUTIONS, True),
        (HistogramType.LINF_SUM_CONTRIBUTIONS, False),
        (HistogramType.COUNT_PER_PARTITION, True),
        (HistogramType.COUNT_PRIVACY_ID_PER_PARTITION, True),
    )
    def test_is_integer(self, name: HistogramType, expected: bool):
        histogram = hist.Histogram(name, bins=[])
        self.assertEqual(histogram.is_integer, expected)

    @parameterized.named_parameters(
        dict(testcase_name='no bins in integer histogram',
             name=HistogramType.L0_CONTRIBUTIONS,
             bins=[],
             expected_lower=None,
             expected_upper=None),
        dict(testcase_name='no bins in floating histogram',
             name=HistogramType.LINF_SUM_CONTRIBUTIONS,
             bins=[],
             expected_lower=None,
             expected_upper=None),
        dict(testcase_name='bins present in integer histogram',
             name=HistogramType.L0_CONTRIBUTIONS,
             bins=[frequency_bin(lower=4, upper=5)],
             expected_lower=1,
             expected_upper=None),
        dict(testcase_name='1 bin in floating histogram',
             name=HistogramType.LINF_SUM_CONTRIBUTIONS,
             bins=[frequency_bin(lower=0.1, upper=0.2)],
             expected_lower=0.1,
             expected_upper=0.2),
        dict(testcase_name='1 bin in floating histogram with the same '
             'lower and upper',
             name=HistogramType.LINF_SUM_CONTRIBUTIONS,
             bins=[frequency_bin(lower=0.1, upper=0.1)],
             expected_lower=0.1,
             expected_upper=0.1),
        dict(testcase_name='multiple bins in floating histogram',
             name=HistogramType.LINF_SUM_CONTRIBUTIONS,
             bins=[
                 frequency_bin(lower=0.1, upper=0.2),
                 frequency_bin(lower=0.3, upper=0.4)
             ],
             expected_lower=0.1,
             expected_upper=0.4),
    )
    def test_lower_and_upper(self, name: HistogramType,
                             bins: List[FrequencyBin],
                             expected_lower: Union[int, float],
                             expected_upper: Union[int, float]):
        histogram = hist.Histogram(name, bins)
        self.assertEqual(histogram.lower, expected_lower)
        self.assertEqual(histogram.upper, expected_upper)


if __name__ == '__main__':
    absltest.main()
