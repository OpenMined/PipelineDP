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
"""Tests for poisson_binomial.py."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from utility_analysis_new import poisson_binomial


class PoissonBinomialTest(parameterized.TestCase):

    @parameterized.parameters(
        ([], [1]), ([1], [0, 1]), ([0, 1], [0, 1, 0]),
        ([0.5] * 4, [0.0625, 0.25, 0.375, 0.25, 0.0625]),
        ([0.1, 0.2, 0.3], [0.504, 0.398, 0.092, 0.006]), ([0.2] * 10, [
            0.107374182, 2.68435456e-01, 3.01989888e-01, 2.01326592e-01,
            8.80803840e-02, 2.64241152e-02, 5.50502400e-03, 7.86432000e-04,
            7.37280000e-05, 4.09600000e-06, 1.02400000e-07
        ]))
    def test_compute_pmf(self, probabilities, expected_pmf):
        pmf = poisson_binomial.compute_pmf(probabilities)
        self.assertSequenceAlmostEqual(expected_pmf, pmf)

    @parameterized.parameters(
        ([0, 0.5], [0.5, 0.5, 0]),
        ([0.5] * 5, [2.5, np.sqrt(1.25), 0]),
        ([0.1, 0.2, 0.3], [0.6, 0.6782329983125269, 0.8077254989355231]),
    )
    def test_compute_exp_std_skewness(self, probabilities, expected_result):
        result = poisson_binomial.compute_exp_std_skewness(probabilities)
        self.assertSequenceAlmostEqual(expected_result, result)

    @parameterized.parameters(([0.5] * 20, 1e-3), ([0.3] * 100, 2e-4),
                              (np.linspace(0.1, 0.9, num=50), 2e-4))
    def test_compute_pmf_approximation(self, probabilities, delta):
        exact_pmf = poisson_binomial.compute_pmf(probabilities)
        mean, std, skewness = poisson_binomial.compute_exp_std_skewness(
            probabilities)
        approximate_pmf = poisson_binomial.compute_pmf_approximation(
            mean, std, skewness, len(probabilities))
        self.assertSequenceAlmostEqual(exact_pmf, approximate_pmf, delta=delta)


if __name__ == '__main__':
    absltest.main()
