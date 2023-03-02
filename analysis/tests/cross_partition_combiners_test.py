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
"""Tests for cross-partition utility analysis combiners."""
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import math

from analysis import metrics
from analysis import cross_partition_combiners
import pipeline_dp


class PerPartitionToCrossPartitionMetrics(parameterized.TestCase):

    @staticmethod
    def get_sum_metrics():
        return metrics.SumMetrics(sum=10.0,
                                  per_partition_error_min=3.0,
                                  per_partition_error_max=-5.0,
                                  expected_cross_partition_error=-2.0,
                                  std_cross_partition_error=3.0,
                                  std_noise=4.0,
                                  noise_kind=pipeline_dp.NoiseKind.LAPLACE)

    def test_metric_utility_count(self):
        input = self.get_sum_metrics()
        output: metrics.MetricUtility = cross_partition_combiners._sum_metrics_to_metric_utility(
            input, pipeline_dp.Metrics.COUNT, partition_keep_probability=1.0)

        self.assertEqual(output.metric, pipeline_dp.Metrics.COUNT)
        self.assertEqual(output.num_dataset_partitions, 1)
        self.assertEqual(output.num_empty_partitions, 0)
        self.assertEqual(output.noise_kind, input.noise_kind)
        self.assertEqual(output.noise_std, input.std_noise)

        # Check absolute_error.
        abs_error: metrics.ValueErrors = output.absolute_error
        self.assertEqual(abs_error.mean, -4)
        self.assertEqual(abs_error.variance, 25)
        self.assertAlmostEqual(abs_error.rmse,
                               math.sqrt(4 * 4 + 25),
                               delta=1e-12)

        bounding_errors = abs_error.bounding_errors
        self.assertEqual(bounding_errors.l0, metrics.MeanVariance(-2.0, 9))
        self.assertEqual(bounding_errors.linf_min, 3.0)
        self.assertEqual(bounding_errors.linf_max, -5.0)

        # Check relative_error.
        expected_rel_error = abs_error.to_relative(input.sum)
        self.assertEqual(output.relative_error, expected_rel_error)
        self.assertAlmostEqual(output.relative_error.mean, -0.4, delta=1e-12)

    def test_metric_utility_empty_partition(self):
        input = self.get_sum_metrics()
        input.sum = 0
        output: metrics.MetricUtility = cross_partition_combiners._sum_metrics_to_metric_utility(
            input, pipeline_dp.Metrics.COUNT, partition_keep_probability=1.0)
        self.assertEqual(output.num_empty_partitions, 1)


# Dataclasses for DataclassHelpersTests
@dataclasses.dataclass
class dataclass1:
    field11: float
    field12: int


@dataclasses.dataclass
class dataclass2:
    field21: float
    field22: dataclass1


class DataclassHelpersTests(parameterized.TestCase):

    def test_add(self):
        input1 = dataclass2(1.0, dataclass1(10.0, 100))
        input2 = dataclass2(2.0, dataclass1(20.0, 200))
        exptected_output = dataclass2(3.0, dataclass1(30.0, 300))
        cross_partition_combiners._add_dataclasses_by_fields(
            input1, input2, fields_to_ignore=[])
        self.assertEqual(input1, exptected_output)

    def test_add_some_fields_ignored(self):
        input1 = dataclass2(1.0, dataclass1(10.0, 100))
        input2 = dataclass2(2.0, dataclass1(20.0, 200))
        exptected_output = dataclass2(3.0, dataclass1(
            30.0, field12=100))  # field12 is ignored
        cross_partition_combiners._add_dataclasses_by_fields(
            input1, input2, fields_to_ignore=["field12"])
        self.assertEqual(input1, exptected_output)

    def test_multiply_float_by_number(self):
        factor = 5
        dataclass_object = dataclass2(1.0, dataclass1(10.0, 100))
        exptected_output = dataclass2(5.0, dataclass1(50.0, 100))
        cross_partition_combiners._multiply_float_dataclasses_field(
            dataclass_object, factor)
        self.assertEqual(dataclass_object, exptected_output)


if __name__ == '__main__':
    absltest.main()
