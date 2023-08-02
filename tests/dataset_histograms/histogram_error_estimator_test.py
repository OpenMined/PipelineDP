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
        # 2nd privacy unit contributes to 1 partition 20 times.
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

    @parameterized.parameters((0, 1), (1, 9 / 11), (2, 8 / 11), (3, 7 / 11),
                              (9, 1 / 11), (10, 0), (20, 0))
    # there are 11 (privacy_id, partition) pairs (from 2 privacy units), when
    # l0_bound=1, 9 are dropped (from 1 privacy unit).
    def test_get_ratio_dropped_l0(self, l0_bound, expected):
        estimator = self._get_estimator(pipeline_dp.Metrics.COUNT)
        self.assertAlmostEqual(estimator.get_ratio_dropped_l0(l0_bound),
                               expected)

    @parameterized.parameters((0, 1), (1, 19 / 30), (2, 18 / 30), (10, 10 / 30),
                              (20, 0), (21, 0))
    # there are 30 rows (from 2 privacy units), when linf_bound=1, 19 are
    # dropped (from 1 privacy unit, which contributes 20 to 1 partition).
    def test_get_ratio_dropped_linf(self, linf_bound, expected):
        estimator = self._get_estimator(pipeline_dp.Metrics.COUNT)
        self.assertAlmostEqual(estimator.get_ratio_dropped_linf(linf_bound),
                               expected)

    @parameterized.parameters((1, 1, 3.9565310998335823),
                              (1, 2, 5.683396971098993),
                              (10, 10, 200.01249625055996))
    # This is explanation how estimation is computed. See _get_histograms
    # for dataset description.
    # l0_bound = linf_bound = 1
    # ratio_dropped_l0 = 9/11, ratio_dropped_linf = 19/30.
    # total_ratio_dropped is estimated as 1 - (1 - 9/11)*(1 - 19/30) ~= 0.933333
    # noise_stddev = 2
    # RMSE is estimated separately on partitions with 1 row and on the partition
    # with 21 rows.
    # On a partition with 1 row (9 such partitions):
    # rmse1 = sqrt(1*total_ratio_dropped + noise_stddev**2) ~= 2.20706
    # On a partition with 21 row:
    # rmse2 = sqrt(21*total_ratio_dropped + noise_stddev**2) ~= 19.70177
    # rmse = (9*rmse1+rmse2)/10.
    def test_estimate_rmse_count(self, l0_bound, linf_bound, expected):
        estimator = self._get_estimator(pipeline_dp.Metrics.COUNT)
        self.assertAlmostEqual(estimator.estimate_rmse(l0_bound, linf_bound),
                               expected)


if __name__ == '__main__':
    absltest.main()
