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

import numpy as np
import hashlib


def choose_from_list_without_replacement(a: list, size: int) -> list:
    if len(a) <= size:
        return a
    # np.random.choice makes casting of elements to numpy types
    # which is undesirable by 2 reasons:
    # 1. Apache Beam can not serialize numpy types.
    # 2. It might lead for losing precision (e.g. arbitrary
    # precision int is converted to int64).
    sampled_indices = np.random.choice(np.arange(len(a)), size, replace=False)

    return [a[i] for i in sampled_indices]


def _compute_64bit_hash(v) -> int:
    m = hashlib.sha1()
    m.update(repr(v).encode())
    return int(m.hexdigest()[:16], 16)


class ValueSampler:
    """Deterministic value sampler.

    For a value it returns whether this value should be kept. The keeping
    decision is deterministic for a fixed value. For a randomly chosen value the
    decision to keep happens with probability sampling_rate.
    """

    def __init__(self, sampling_rate: float):
        self._sample_bound = int(round(2**64 * sampling_rate))

    def keep(self, value) -> bool:
        """Returns true if the value should be kept."""
        return _compute_64bit_hash(value) < self._sample_bound
