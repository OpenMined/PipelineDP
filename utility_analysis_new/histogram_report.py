# Copyright 2022 OpenMined.
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
"""Histogram reports for utility analysis."""
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt


class HistogramReport:
    """Histogram report stores entries for utility analysis and represents them in a graph."""

    def __init__(self, data_pairs: List[Tuple[int, int]]):
        self.data_pairs = data_pairs
        self._user_contrib_freq_bins: Dict[int, int] = {}

    def plot_partition_contrib_freq(self):
        plt.bar(self._user_contrib_freq_bins.keys(),
                self._user_contrib_freq_bins.values())
        plt.show()

    def calc_partition_contrib_freq(self) -> dict:
        """Rearranges the contents of the stored data into contribution frequency histogram."""
        if not self._user_contrib_freq_bins:
            for pair in self.data_pairs:
                self._user_contrib_freq_bins[
                    pair[1]] = self._user_contrib_freq_bins.get(pair[1], 0) + 1

        return self._user_contrib_freq_bins

    def __str__(self):
        report_str = "Histogram Report"
        for k, v in self._user_contrib_freq_bins:
            report_str += f" {k}, {v} \n"
        return report_str
