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
"""TODO."""
import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from typing import Optional, Sequence, Tuple
import math


class CountErrorEstimator:

    def __init__(self, histograms: hist.DatasetHistograms, base_std: float,
                 metric: pipeline_dp.Metrics, noise: pipeline_dp.NoiseKind):
        # TODO: assert COUNT, PRIVACY_ID_COUNT
        self._l0_ratios_dropped = hist.compute_ratio_dropped(
            histograms.l0_contributions_histogram)
        self._linf_ratios_dropped = hist.compute_ratio_dropped(
            histograms.linf_contributions_histogram)
        self._histograms = histograms
        self._base_std = base_std
        self._metric = metric
        self._num_partitions = histograms.count_per_partition_histogram.total_count(
        )

    def estimate_rmse(self, l0: int, linf: Optional[int] = None) -> float:
        # todo validate linf
        ratio_dropped_l0 = self._get_ratio_dropped(self._l0_ratios_dropped, l0)
        ratio_dropped_linf = 0
        if self._metric == pipeline_dp.Metrics.COUNT:
            ratio_dropped_linf = self._get_ratio_dropped(
                self._linf_ratios_dropped, linf)
            partition_histogram = self._histograms.count_per_partition_histogram
        else:
            partition_histogram = self._histograms.count_privacy_id_per_partition
        ratio_dropped = 1 - (1 - ratio_dropped_l0) * (1 - ratio_dropped_linf)
        sigma = self._get_sigma(l0, linf)
        sum_rmse = 0
        for bin in partition_histogram.bins:
            average_per_bin = bin.sum / bin.count
            rmse = math.sqrt((ratio_dropped * average_per_bin)**2 + sigma**2)
            sum_rmse += bin.count * rmse
        return sum_rmse / self._num_partitions

    def _get_ratio_dropped(self, ratios_dropped: Sequence[Tuple[int, float]],
                           bound: int) -> float:
        pass

    def _get_sigma(self, l0: int, linf: Optional[int] = None) -> float:
        pass
