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

from absl.testing import absltest
from absl.testing import parameterized
from typing import Sequence

from utility_analysis_new import probability_computations


class ProbabilityComputationsTest(parameterized.TestCase):

    # Expected quantiles are computed with the analytical approach with
    # precision 1e-10. Unfortunetely the analytical apprach is too slow to be
    # used in production.
    @parameterized.parameters(
        (1.0, 2.0, [0.1, 0.5, 0.9], [-3.08740234375, 0, 3.08740234375]),
        (1.01, 0.55, [0.5, 0.6, 0.7, 0.8, 0.9, 0.99], [
            0, 0.302294921875, 0.638916015625, 1.069970703125, 1.775146484375,
            4.100927734375
        ]),
    )
    def test_compute_sum_laplace_gaussian_quantiles(
            self, laplace_b: float, gaussian_sigma: float,
            quantiles_to_compute: Sequence[float],
            expected_quantiles: Sequence[float]):
        computed_quantiles = probability_computations.compute_sum_laplace_gaussian_quantiles(
            laplace_b, gaussian_sigma, quantiles_to_compute, 4 * 10**6)
        self.assertSequenceAlmostEqual(expected_quantiles,
                                       computed_quantiles,
                                       delta=0.01)


if __name__ == '__main__':
    absltest.main()
