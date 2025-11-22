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

from typing import Optional

from pipeline_dp import aggregate_params as ap
from pipeline_dp import data_extractors as de
from pipeline_dp import pipeline_backend
from pipeline_dp.pipeline_backend import LocalBackend
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp.dataset_histograms import computing_histograms
from pipeline_dp.dataset_histograms import histogram_error_estimator


class HistogramErrorEstimatorTest(parameterized.TestCase):

    def _get_histograms(self) -> hist.DatasetHistograms:
        # Generate dataset
        dataset = []
        # 1st privacy unit contributes to 10 partitions once
        dataset.extend([(1, i, 1.0) for i in range(10)])
        # 2nd privacy unit contributes to 1 partition 20 times.
        dataset.extend([(2, 0, 2.0) for i in range(20)])

        data_extractors = de.DataExtractors(privacy_id_extractor=lambda x: x[0],
                                            partition_extractor=lambda x: x[1],
                                            value_extractor=lambda x: x[2])
        return list(
            computing_histograms.compute_dataset_histograms(
                dataset, data_extractors, pipeline_backend.LocalBackend()))[0]

    def _get_estimator_for_count_and_privacy_id_count(
        self,
        metric: ap.Metric,
        noise_kind: ap.NoiseKind = ap.NoiseKind.LAPLACE,
        epsilon: float = 2**0.5 / 2,
        delta: Optional[float] = None,
    ):
        return histogram_error_estimator.create_estimator_for_count_and_privacy_id_count(
            self._get_histograms(), epsilon, delta, metric, noise_kind)

    def _get_estimator_for_sum(
        self,
        noise_kind: ap.NoiseKind = ap.NoiseKind.LAPLACE,
        epsilon: float = 2**0.5 / 2,
        delta: Optional[float] = None,
    ):
        return histogram_error_estimator.create_estimator_for_sum(
            self._get_histograms(), epsilon, delta, noise_kind)

    @parameterized.named_parameters(
        dict(testcase_name='count_gaussian',
             metric=ap.Metrics.COUNT,
             noise_kind=ap.NoiseKind.GAUSSIAN,
             epsilon=1.0,
             delta=0.00685,
             l0=9,
             linf=5,
             expected=30),
        dict(testcase_name='count_laplace',
             metric=ap.Metrics.COUNT,
             noise_kind=ap.NoiseKind.LAPLACE,
             epsilon=2**0.5 / 2,
             delta=None,
             l0=9,
             linf=5,
             expected=90),
        dict(testcase_name='privacy_id_count_gaussian',
             metric=ap.Metrics.PRIVACY_ID_COUNT,
             noise_kind=ap.NoiseKind.GAUSSIAN,
             epsilon=1.0,
             delta=0.031,
             l0=9,
             linf=5,
             expected=4.5),
        dict(testcase_name='privacy_id_count_laplace',
             metric=ap.Metrics.PRIVACY_ID_COUNT,
             noise_kind=ap.NoiseKind.LAPLACE,
             epsilon=2**0.5 / 1.5,
             delta=None,
             l0=9,
             linf=5,
             expected=13.5),
    )
    def test_count_get_sigma(self, metric: ap.Metric, epsilon: float,
                             delta: Optional[float], noise_kind: ap.NoiseKind,
                             l0: float, linf: float, expected: float):
        estimator = self._get_estimator_for_count_and_privacy_id_count(
            metric=metric, epsilon=epsilon, delta=delta, noise_kind=noise_kind)
        self.assertAlmostEqual(estimator._get_stddev(l0, linf),
                               expected,
                               delta=1e-10)

    def test_sum_not_supported(self):
        with self.assertRaisesRegex(
                ValueError, "Only COUNT and PRIVACY_ID_COUNT are supported"):
            self._get_estimator_for_count_and_privacy_id_count(ap.Metrics.SUM)

    @parameterized.parameters((0, 1), (1, 9 / 11), (2, 8 / 11), (3, 7 / 11),
                              (9, 1 / 11), (10, 0), (20, 0))
    # there are 11 (privacy_id, partition) pairs (from 2 privacy units), when
    # l0_bound=1, 9 are dropped (from 1 privacy unit).
    def test_get_ratio_dropped_l0(self, l0_bound, expected):
        estimator = self._get_estimator_for_count_and_privacy_id_count(
            ap.Metrics.COUNT)
        self.assertAlmostEqual(estimator.get_ratio_dropped_l0(l0_bound),
                               expected)

    @parameterized.parameters((0, 1), (1, 9 / 11), (2, 8 / 11), (3, 7 / 11),
                              (9, 1 / 11), (10, 0), (20, 0))
    # there are 11 (privacy_id, partition) pairs (from 2 privacy units), when
    # l0_bound=1, 9 are dropped (from 1 privacy unit).
    def test_get_ratio_dropped_l0_for_sum(self, l0_bound, expected):
        estimator = self._get_estimator_for_sum()
        self.assertAlmostEqual(estimator.get_ratio_dropped_l0(l0_bound),
                               expected)

    @parameterized.parameters((0, 1), (1, 19 / 30), (2, 18 / 30), (10, 10 / 30),
                              (20, 0), (21, 0))
    # there are 30 rows (from 2 privacy units), when linf_bound=1, 19 are
    # dropped (from 1 privacy unit, which contributes 20 to 1 partition).
    def test_get_ratio_dropped_linf(self, linf_bound, expected):
        estimator = self._get_estimator_for_count_and_privacy_id_count(
            ap.Metrics.COUNT)
        self.assertAlmostEqual(estimator.get_ratio_dropped_linf(linf_bound),
                               expected)

    @parameterized.parameters((0, 1), (0.5, 0.89), (1, 0.78), (2, 0.76),
                              (40, 0))
    # there 1 is contribution of 40 and 10 contribution of 1.
    # total contribution = 1*40+10*1 = 50
    # when linf_bound = 0.5, left after contribution bounding 11*0.5=5.5, i.e.
    # dropped (50-5.5)/50 = 0.89
    def test_get_ratio_dropped_linf_for_sum(self, linf_bound, expected):
        estimator = self._get_estimator_for_sum()
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
        estimator = self._get_estimator_for_count_and_privacy_id_count(
            ap.Metrics.COUNT)
        self.assertAlmostEqual(estimator.estimate_rmse(l0_bound, linf_bound),
                               expected)

    def test_estimate_rmse_sum(self):
        estimator = self._get_estimator_for_sum()
        self.assertAlmostEqual(estimator.estimate_rmse(1, 1), 5.93769917)


if __name__ == '__main__':
    absltest.main()
