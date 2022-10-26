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
"""Computations related to probabilistic distributions."""

import numpy as np
from typing import List, Sequence


def compute_sum_laplace_gaussian_quantiles(laplace_b: float,
                                           gaussian_sigma: float,
                                           quantiles: Sequence[float],
                                           num_samples: int) -> List[float]:
    """Computes quantiles for the sum of independent Laplace and Gaussian distributions."""
    # There are exact formulas for computing Laplace+Gaussian cdf, but
    # it turned out that their Python implementation is too slow.
    # That is why the Monte-Carlo method is used. The Monte-Carlo method is also
    # pretty slow.
    # num_samples = 10**3, 4500 calls/sec
    # num_samples = 10**4, 800 calls/sec

    samples = np.random.laplace(
        scale=laplace_b, size=num_samples) + np.random.normal(
            loc=0, scale=gaussian_sigma, size=num_samples)
    return np.quantile(samples, quantiles)
