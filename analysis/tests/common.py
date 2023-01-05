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
"""Common functions for utility analysis tests"""

import dataclasses
import enum
import typing

from absl.testing import parameterized


def _assert_dataclasses_are_equal(test: parameterized.TestCase,
                                  expected,
                                  actual,
                                  delta=1e-5):
    """Asserts that input dataclasses are equal to one another.

  Only supports dataclasses with the following fields:
  - int
  - float
  - List[int]
  - List[float]
  - enum

  For floats, it uses approximate equality with given delta (defaults to 1e-5).
  """
    test.assertEquals(type(expected),
                      type(actual),
                      msg=f"expected={type(expected)} and actual={type(actual)}"
                      f"need to be the same type")
    expected = dataclasses.asdict(expected)
    actual = dataclasses.asdict(actual)

    for field_name in actual.keys():
        exp = expected[field_name]
        act = actual[field_name]
        if isinstance(act, int) or isinstance(act, enum.Enum):
            test.assertEquals(
                exp,
                act,
                msg=f"expected={exp} and actual={act} differ in {field_name}")
        elif isinstance(act, float):
            test.assertAlmostEquals(
                exp,
                act,
                delta=delta,
                msg=f"expected={exp} and actual={act} differ in {field_name}")
        elif isinstance(act, typing.List):
            [
                test.assertAlmostEquals(
                    exp_i,
                    act_i,
                    delta=delta,
                    msg=f"expected={exp_i} and actual={act_i} differ in "
                    f"{field_name} at index= {i}")
                for i, (exp_i, act_i) in enumerate(zip(exp, act))
            ]
        else:
            raise Exception(
                f"assert_dataclasses_are_equal only supports dataclasses with "
                f"int, float, List[int], List[float] or enum fields. Got "
                f"f1={field_name} with type={type(act)} instead")
