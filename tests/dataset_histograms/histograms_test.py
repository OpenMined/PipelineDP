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

from absl.testing import absltest
from absl.testing import parameterized

from pipeline_dp.dataset_histograms import histograms as hist


class HistogramTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='1 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1000, count=10, sum=10100, max=1009),
             ],
             q=[0.05, 0.1, 0.5, 0.8, 0.9],
             expected_quantiles=[1000, 1000, 1000, 1000, 1000]),
        dict(testcase_name='6 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1, count=2, sum=2, max=1),
                 hist.FrequencyBin(lower=2, count=1, sum=2, max=2),
                 hist.FrequencyBin(lower=3, count=1, sum=3, max=3),
                 hist.FrequencyBin(lower=4, count=2, sum=8, max=4),
                 hist.FrequencyBin(lower=5, count=2, sum=10, max=5),
                 hist.FrequencyBin(lower=6, count=1, sum=6, max=6),
                 hist.FrequencyBin(lower=10, count=1, sum=11, max=11)
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
                 hist.FrequencyBin(lower=1000, count=10, sum=10100, max=1020),
             ],
             expected_ratios=[(0, 1), (1000, 100 / 10100), (1020, 0.0)]),
        dict(testcase_name='7 bins histogram',
             bins=[
                 hist.FrequencyBin(lower=1, count=8, sum=8, max=1),
                 hist.FrequencyBin(lower=2, count=2, sum=4, max=2),
                 hist.FrequencyBin(lower=3, count=1, sum=3, max=3),
                 hist.FrequencyBin(lower=4, count=2, sum=8, max=4),
                 hist.FrequencyBin(lower=5, count=2, sum=10, max=5),
                 hist.FrequencyBin(lower=6, count=1, sum=6, max=6),
                 hist.FrequencyBin(lower=11, count=1, sum=11, max=11),
             ],
             expected_ratios=[(0, 1), (1, 0.66), (2, 0.48), (3, 0.34),
                              (4, 0.22), (5, 0.14), (6, 0.1), (11, 0.0)]))
    def test_ratio_dropped(self, bins, expected_ratios):
        histogram = hist.Histogram("name", bins)
        output = hist.compute_ratio_dropped(histogram)
        self.assertListEqual(output, expected_ratios)


if __name__ == '__main__':
    absltest.main()
