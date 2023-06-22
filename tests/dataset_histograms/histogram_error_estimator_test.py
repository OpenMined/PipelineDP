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
"""Tests for histogram error estimator."""

from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import computing_histograms
from pipeline_dp.dataset_histograms import histogram_error_estimator


class HistogramErrorEstimatorTest(parameterized.TestCase):

    def _get_histograms(self) -> hist.DatasetHistograms:
        # Generate dataset
        dataset = []
        # 1st privacy unit contributes to 10 partitions once
        dataset.extend([(1, i) for i in range(10)])
        # 2st privacy unit contributes to 1 partition 20 times.
        dataset.extend([(2, 0) for i in range(20)])

        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1])
        return list(
            computing_histograms.compute_dataset_histograms(
                dataset, data_extractors, pipeline_dp.LocalBackend()))[0]

    def _get_estimator(
            self,
            metric: pipeline_dp.Metric,
            noise_kind: pipeline_dp.NoiseKind = pipeline_dp.NoiseKind.LAPLACE,
            base_std: float = 2.0):
        return histogram_error_estimator.create_error_estimator(
            self._get_histograms(), base_std, metric, noise_kind)

    @parameterized.named_parameters(
        dict(testcase_name='count_gaussian',
             metric=pipeline_dp.Metrics.COUNT,
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             base_std=2.0,
             l0=9,
             linf=5,
             expected=30),
        dict(testcase_name='count_laplace',
             metric=pipeline_dp.Metrics.COUNT,
             noise_kind=pipeline_dp.NoiseKind.LAPLACE,
             base_std=2.0,
             l0=9,
             linf=5,
             expected=90),
        dict(testcase_name='privacy_id_count_gaussian',
             metric=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             base_std=1.5,
             l0=9,
             linf=5,
             expected=4.5),
        dict(testcase_name='privacy_id_count_laplace',
             metric=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
             noise_kind=pipeline_dp.NoiseKind.LAPLACE,
             base_std=1.5,
             l0=9,
             linf=5,
             expected=13.5),
    )
    def test_count_get_sigma(self, metric: pipeline_dp.Metric, base_std: float,
                             noise_kind: pipeline_dp.NoiseKind, l0: float,
                             linf: float, expected: float):
        estimator = self._get_estimator(metric=metric,
                                        base_std=base_std,
                                        noise_kind=noise_kind)
        self.assertAlmostEqual(estimator._get_stddev(l0, linf),
                               expected,
                               delta=1e-10)

    def test_sum_not_supported(self):
        with self.assertRaisesRegex(
                ValueError, "Only COUNT and PRIVACY_ID_COUNT are supported"):
            self._get_estimator(pipeline_dp.Metrics.SUM)

    def test_get_ratio_dropped_l0(self):
        estimator = self._get_estimator(pipeline_dp.Metrics.COUNT)
        self.assertAlmostEqual(estimator.get_ratio_dropped_l0(1), 0)


if __name__ == '__main__':
    absltest.main()
