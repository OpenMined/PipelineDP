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


def assert_dataclasses_are_equal(test: parameterized.TestCase,
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
    test.assertEqual(type(expected),
                     type(actual),
                     msg=f"expected={type(expected)} and actual={type(actual)}"
                     f"need to be the same type")
    expected = dataclasses.asdict(expected)
    actual = dataclasses.asdict(actual)
    assert_dictionaries_are_equal(test, expected, actual, delta)


def assert_fields_are_equal(
        test: parameterized.TestCase,
        expected,  # todo rename
        actual,
        field_name: str,
        delta=1e-5):
    if expected == actual:
        return
    if expected is None:
        test.assertIsNone(actual, msg=f"{field_name} is expected tp be None.")
    elif isinstance(expected, int) or isinstance(expected, enum.Enum):
        test.assertEquals(
            expected,
            actual,
            msg=f"expected={expected} and actual={actual} differ in {field_name}"
        )
    elif isinstance(expected, float):
        test.assertAlmostEqual(
            expected,
            actual,
            delta=delta,
            msg=f"expected={expected} and actual={actual} differ in {field_name}"
        )
    elif isinstance(expected, typing.List):
        for i, (exp_i, act_i) in enumerate(zip(expected, actual)):
            assert_fields_are_equal(test, exp_i, act_i, f"{field_name}[{i}]",
                                    delta)
    elif isinstance(expected, typing.Dict):
        assert_dictionaries_are_equal(test, expected, actual, delta)
    else:
        raise Exception(
            f"assert_dataclasses_are_equal only supports dataclasses with "
            f"int, float, List[int], List[float] or enum fields. Got "
            f"f1={field_name} with type={type(actual)} instead")


def assert_dictionaries_are_equal(test: parameterized.TestCase,
                                  expected,
                                  actual,
                                  delta=1e-5):
    for field_name in expected.keys():
        assert_fields_are_equal(test, expected[field_name], actual[field_name],
                                field_name, delta)
