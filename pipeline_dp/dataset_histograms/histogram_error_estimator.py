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
"""Estimation of errors from DatasetHistograms."""
import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from pydp.algorithms import numerical_mechanisms as dp_mechanisms
from typing import Optional, Sequence, Tuple
import math
import bisect


class ErrorEstimator:
    """Estimator of the error from DP pipeline from DatasetHistograms.

    The recommended way to create this object is to use create_error_estimator.

    Partition selection error is not implemented yet. Now only contribution
    bounding and noise error are taken into consideration.
    """

    def __init__(
        self,
        epsilon: float,
        delta: Optional[float],
        metric: pipeline_dp.Metric,
        noise: pipeline_dp.NoiseKind,
        l0_ratios_dropped: Sequence[Tuple[int, float]],
        linf_ratios_dropped: Sequence[Tuple[int, float]],
        partition_histogram: hist.Histogram,
    ):
        self._base_std = self._get_stddev_for_dp_mechanism(
            epsilon, delta, noise)
        self._metric = metric
        self._noise = noise
        self._l0_ratios_dropped = l0_ratios_dropped
        self._linf_ratios_dropped = linf_ratios_dropped
        self._partition_histogram = partition_histogram

    def _get_stddev_for_dp_mechanism(
        self,
        epsilon: float,
        delta: Optional[float],
        noise: pipeline_dp.NoiseKind,
    ) -> float:
        if noise == pipeline_dp.NoiseKind.LAPLACE:
            if delta is not None:
                raise ValueError("Delta must be None for Laplace noise")
            return 2**0.5 / epsilon
        elif noise == pipeline_dp.NoiseKind.GAUSSIAN:
            return dp_mechanisms.GaussianMechanism(epsilon=epsilon,
                                                   delta=delta,
                                                   sensitivity=1.0).std
        else:
            raise ValueError(f"Unsupported noise kind: {noise}")

    def estimate_rmse(self,
                      l0_bound: int,
                      linf_bound: Optional[int] = None,
                      to_print=False) -> float:
        """Estimates RMSE error for given l0 and linf bounds.

        Estimation algorithm is the following:
        1. From l0_bound and l0_contributions_histogram the ratio data dropped
          from l0 contribution bounding is computed.
        2. From linf_bound and linf_contributions_histogram the
          ratio data dropped from linf contribution bounding is computed.
        3. The total 'ratio_data_dropped' for contribution bounding is estimated
          from l0 and linf ratios data dropped.
        4. Then under the assumption that contribution bounding drops data
          uniformly on all partitions, for a partition of the size n, it is
          assumed that n*ratio_data_dropped data points are dropped with
          contribution bounding. And RMSE for this partition is computed as
          sqrt((n*ratio_data_dropped)**2 + noise_std**2)
        5. RMSE are averaged across all partitions.

        Args:
            l0_bound: l0 contribution bound, AKA max_partition_contributed.
            linf_bound: linf contribution bound, AKA for COUNT as
              max_contributions_per_partition. This parameter is ignored for
              PRIVACY_ID_COUNT

        Returns:
            the estimated error.
        """
        if self._metric != pipeline_dp.Metrics.PRIVACY_ID_COUNT:
            if linf_bound is None:
                raise ValueError("linf must be given for COUNT")
        ratio_dropped_l0 = self.get_ratio_dropped_l0(l0_bound)
        ratio_dropped_linf = 0
        if self._metric != pipeline_dp.Metrics.PRIVACY_ID_COUNT:
            ratio_dropped_linf = self.get_ratio_dropped_linf(linf_bound)
        ratio_dropped = 1 - (1 - ratio_dropped_l0) * (1 - ratio_dropped_linf)
        stddev = self._get_stddev(l0_bound, linf_bound)
        if to_print:
            print(f"{ratio_dropped_l0=},"
                  f" {ratio_dropped_linf=} {ratio_dropped=} {stddev=}")
        return _estimate_rmse_impl(ratio_dropped, stddev,
                                   self._partition_histogram)

    def get_ratio_dropped_l0(self, l0_bound: int) -> float:
        """Computes ratio"""
        return self._get_ratio_dropped(self._l0_ratios_dropped, l0_bound)

    def get_ratio_dropped_linf(self, linf_bound: int) -> float:
        return self._get_ratio_dropped(self._linf_ratios_dropped, linf_bound)

    def _get_ratio_dropped(self, ratios_dropped: Sequence[Tuple[int, float]],
                           bound: int) -> float:
        if bound <= 0:
            return 1
        if bound > ratios_dropped[-1][0]:
            return 0
        index = bisect.bisect_left(ratios_dropped, (bound, 0))
        if ratios_dropped[index][0] == bound:
            return ratios_dropped[index][1]

        # index > 0, because ratio_dropped starts from 0, and bound > 0.
        x1, y1 = ratios_dropped[index - 1]
        x2, y2 = ratios_dropped[index]
        # Linearly interpolate between (x1, y1) and (x2, y2) for x=bound.
        return (y1 * (x2 - bound) + y2 * (bound - x1)) / (x2 - x1)

    def _get_stddev(self,
                    l0_bound: int,
                    linf_bound: Optional[int] = None) -> float:
        if self._metric == pipeline_dp.Metrics.PRIVACY_ID_COUNT:
            linf_bound = 1
        if self._noise == pipeline_dp.NoiseKind.LAPLACE:
            return self._base_std * l0_bound * linf_bound
        # Gaussian noise case.
        return self._base_std * math.sqrt(l0_bound) * linf_bound


def create_estimator_for_count_privacy_id_count(
    histograms: hist.DatasetHistograms,
    epsilon: float,
    delta: Optional[float],
    metric: pipeline_dp.Metric,
    noise: pipeline_dp.NoiseKind,
) -> ErrorEstimator:
    """Creates histogram based error estimator for COUNT or PRIVACY_ID_COUNT.

    Args:
        histograms: dataset histograms.
        epsilon: epsilon parameter of the DP mechanism for adding noise.
        delta: delta parameter of the DP mechanism for adding noise (must be None
          for Laplace noise).
        metric: DP aggregation, COUNT or PRIVACY_ID_COUNT.
        noise: type of DP noise.

    Returns:
        Error estimator.
    """
    if metric not in [
            pipeline_dp.Metrics.COUNT,
            pipeline_dp.Metrics.PRIVACY_ID_COUNT,
    ]:
        raise ValueError(
            f"Only COUNT and PRIVACY_ID_COUNT are supported, but metric={metric}"
        )
    l0_ratios_dropped = hist.compute_ratio_dropped(
        histograms.l0_contributions_histogram)
    linf_ratios_dropped = hist.compute_ratio_dropped(
        histograms.linf_contributions_histogram)
    if metric == pipeline_dp.Metrics.COUNT:
        partition_histogram = histograms.count_per_partition_histogram
    else:
        partition_histogram = histograms.count_privacy_id_per_partition
    return ErrorEstimator(
        epsilon,
        delta,
        metric,
        noise,
        l0_ratios_dropped,
        linf_ratios_dropped,
        partition_histogram,
    )


def _estimate_rmse_impl(ratio_dropped: float, std: float,
                        partition_histogram: hist.Histogram) -> float:
    sum_rmse = 0
    num_partitions = partition_histogram.total_count()
    for bin in partition_histogram.bins:
        average_partition_size_in_bin = bin.sum / bin.count
        rmse = math.sqrt((ratio_dropped * average_partition_size_in_bin)**2 +
                         std**2)
        sum_rmse += bin.count * rmse
    return sum_rmse / num_partitions


def create_estimator_for_sum(histograms: hist.DatasetHistograms,
                             epsilon: float,
                             delta: Optional[float],
                             noise: pipeline_dp.NoiseKind,
                             sum_index: int = 0) -> ErrorEstimator:
    """Creates histogram based error estimator for SUM.

    Args:
        histograms: dataset histograms.
        epsilon: epsilon parameter of the DP mechanism for adding noise.
        delta: delta parameter of the DP mechanism for adding noise (must be None
          for Laplace noise).
        noise: type of DP noise.
        sum_index: the index of the sum for the case of multi-aggregations.

    Returns:
        Error estimator.
    """
    l0_ratios_dropped = hist.compute_ratio_dropped(
        histograms.l0_contributions_histogram)
    if isinstance(histograms.linf_sum_contributions_histogram, hist.Histogram):
        # 1 sum
        linf_sum_histograms = histograms.linf_sum_contributions_histogram
        partition_histogram = histograms.sum_per_partition_histogram
    else:  # multiple SUM aggregations
        linf_sum_histograms = histograms.linf_sum_contributions_histogram[
            sum_index]
        partition_histogram = histograms.sum_per_partition_histogram[sum_index]

    linf_ratios_dropped = hist.compute_ratio_dropped(linf_sum_histograms)

    return ErrorEstimator(
        epsilon,
        delta,
        pipeline_dp.Metrics.SUM,
        noise,
        l0_ratios_dropped,
        linf_ratios_dropped,
        partition_histogram,
    )
