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
from absl.testing import absltest
from absl.testing import parameterized
import dataclasses
import math
import unittest.mock
from unittest.mock import patch

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

    def test_partition_selection_per_to_cross_partition(self):
        output = cross_partition_combiners._partition_selection_per_to_cross_partition(
            0.25)
        self.assertEqual(output.num_partitions, 1)
        self.assertEqual(output.dropped_partitions,
                         metrics.MeanVariance(0.25, 0.25 * 0.75))

    @parameterized.parameters(False, True)
    @patch(
        "analysis.cross_partition_combiners._partition_selection_per_to_cross_partition"
    )
    @patch("analysis.cross_partition_combiners._sum_metrics_to_metric_utility")
    def test_per_partition_to_cross_partition_utility(
            self, public_partitions: bool, mock_sum_metrics_to_metric_utility,
            mock_partition_selection_per_to_cross_partition):
        per_partition_utility = metrics.PerPartitionMetrics(
            0.2, metric_errors=[self.get_sum_metrics(),
                                self.get_sum_metrics()])
        dp_metrics = [
            pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT
        ]
        output = cross_partition_combiners._per_partition_to_cross_partition_utility(
            per_partition_utility, dp_metrics, public_partitions)
        if public_partitions:
            mock_partition_selection_per_to_cross_partition.assert_not_called()
        else:
            mock_partition_selection_per_to_cross_partition.assert_called_once_with(
                0.2)

        self.assertEqual(mock_sum_metrics_to_metric_utility.call_count, 2)

    @patch(
        "analysis.cross_partition_combiners._partition_selection_per_to_cross_partition"
    )
    @patch("analysis.cross_partition_combiners._sum_metrics_to_metric_utility")
    def test_per_partition_to_cross_partition_utility_only_partition_selection(
            self, mock_sum_metrics_to_metric_utility,
            mock_partition_selection_per_to_cross_partition):
        per_partition_utility = metrics.PerPartitionMetrics(0.5,
                                                            metric_errors=None)
        output = cross_partition_combiners._per_partition_to_cross_partition_utility(
            per_partition_utility, [], public_partitions=False)

        self.assertIsNone(output.metric)
        self.assertIsInstance(output.partition_selection,
                              unittest.mock.MagicMock)
        mock_sum_metrics_to_metric_utility.assert_not_called()
        mock_partition_selection_per_to_cross_partition.assert_called_once_with(
            0.5)


# Dataclasses for DataclassHelpersTests
@dataclasses.dataclass
class OuterClass:
    field11: float
    field12: int


@dataclasses.dataclass
class InnerClass:
    field21: float
    field22: OuterClass


class DataclassHelpersTests(parameterized.TestCase):

    def test_add(self):
        input1 = InnerClass(1.0, OuterClass(10.0, 100))
        input2 = InnerClass(2.0, OuterClass(20.0, 200))
        exptected_output = InnerClass(3.0, OuterClass(30.0, 300))
        cross_partition_combiners._add_dataclasses_by_fields(
            input1, input2, fields_to_ignore=[])
        self.assertEqual(input1, exptected_output)

    def test_add_some_fields_ignored(self):
        input1 = InnerClass(1.0, OuterClass(10.0, 100))
        input2 = InnerClass(2.0, OuterClass(20.0, 200))
        exptected_output = InnerClass(3.0, OuterClass(
            30.0, field12=100))  # field12 is ignored
        cross_partition_combiners._add_dataclasses_by_fields(
            input1, input2, fields_to_ignore=["field12"])
        self.assertEqual(input1, exptected_output)

    def test_multiply_float_by_number(self):
        factor = 5
        dataclass_object = InnerClass(1.0, OuterClass(10.0, 100))
        exptected_output = InnerClass(5.0, OuterClass(50.0, 100))
        cross_partition_combiners._multiply_float_dataclasses_field(
            dataclass_object, factor)
        self.assertEqual(dataclass_object, exptected_output)


if __name__ == '__main__':
    absltest.main()
