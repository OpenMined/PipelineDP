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


class CountErrorEstimator:

    def __init__(self, histograms: hist.DatasetHistograms, base_std: float,
                 metric: pipeline_dp.Metrics):
        # TODO: assert COUNT, PRIVACY_ID_COUNT
        self._l0_ratios_dropped = hist.compute_ratio_dropped(
            histograms.l0_contributions_histogram)
        self._linf_ratios_dropped = hist.compute_ratio_dropped(
            histograms.linf_contributions_histogram)
        self._histograms = histograms
        self._base_std = base_std

    def estimate_rmse(self, l0: int, linf: int) -> float:
        ratio_dropped_l0 = self._get_ratio_dropped(self._l0_ratios_dropped, l0)
        ratio_dropped_linf = 0

    def _get_ratio_dropped(self, ratios_dropped: Sequence[Tuple[int, float]],
                           bound: int):
        pass
