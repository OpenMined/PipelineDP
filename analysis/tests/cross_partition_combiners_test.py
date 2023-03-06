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


def _get_sum_metrics():
    return metrics.SumMetrics(sum=10.0,
                              per_partition_error_min=3.0,
                              per_partition_error_max=-5.0,
                              expected_cross_partition_error=-2.0,
                              std_cross_partition_error=3.0,
                              std_noise=4.0,
                              noise_kind=pipeline_dp.NoiseKind.LAPLACE)


class PerPartitionToCrossPartitionMetrics(parameterized.TestCase):

    def test_metric_utility_count(self):
        input = _get_sum_metrics()
        output: metrics.MetricUtility = cross_partition_combiners._sum_metrics_to_metric_utility(
            input, pipeline_dp.Metrics.COUNT, partition_keep_probability=1.0)

        self.assertEqual(output.metric, pipeline_dp.Metrics.COUNT)
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

    @parameterized.parameters(False, True)
    def test_create_partition_metrics_for_public_partitions(
            self, is_empty_partition):
        output: metrics.PartitionMetrics = cross_partition_combiners._partition_metrics_public_partitions(
            is_empty_partition)
        self.assertTrue(output.public_partitions)
        self.assertEqual(output.num_non_public_partitions, 0)
        self.assertEqual(output.num_dataset_partitions,
                         0 if is_empty_partition else 1)
        self.assertEqual(output.num_empty_partitions,
                         1 if is_empty_partition else 0)

    def test_partition_selection_per_to_cross_partition(self):
        output = cross_partition_combiners._partition_metrics_private_partitions(
            0.25)
        self.assertFalse(output.public_partitions)
        self.assertEqual(output.num_dataset_partitions, 1)
        self.assertEqual(output.kept_partitions,
                         metrics.MeanVariance(0.25, 0.25 * 0.75))

    @parameterized.parameters(False, True)
    @patch(
        "analysis.cross_partition_combiners._partition_metrics_public_partitions"
    )
    @patch(
        "analysis.cross_partition_combiners._partition_metrics_private_partitions"
    )
    @patch("analysis.cross_partition_combiners._sum_metrics_to_metric_utility")
    def test_per_partition_to_cross_partition_utility(
            self, public_partitions: bool, mock_sum_metrics_to_metric_utility,
            mock_create_for_private_partitions,
            mock_create_for_public_partitions):
        per_partition_utility = metrics.PerPartitionMetrics(
            0.2, metric_errors=[_get_sum_metrics(),
                                _get_sum_metrics()])
        dp_metrics = [
            pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT
        ]
        cross_partition_combiners._per_partition_to_utility_report(
            per_partition_utility, dp_metrics, public_partitions)
        if public_partitions:
            mock_create_for_public_partitions.assert_called_once_with(False)
            mock_create_for_private_partitions.assert_not_called()
        else:
            mock_create_for_public_partitions.assert_not_called()
            mock_create_for_private_partitions.assert_called_once_with(0.2)

        self.assertEqual(mock_sum_metrics_to_metric_utility.call_count, 2)

    @patch(
        "analysis.cross_partition_combiners._partition_metrics_public_partitions"
    )
    @patch(
        "analysis.cross_partition_combiners._partition_metrics_private_partitions"
    )
    @patch("analysis.cross_partition_combiners._sum_metrics_to_metric_utility")
    def test_per_partition_to_cross_partition_utility_only_partition_selection(
            self, mock_to_metric_utility, mock_create_for_private_partitions,
            mock_create_for_public_partitions):
        per_partition_utility = metrics.PerPartitionMetrics(0.5,
                                                            metric_errors=None)
        output = cross_partition_combiners._per_partition_to_utility_report(
            per_partition_utility, [], public_partitions=False)

        self.assertIsNone(output.metric_errors)
        self.assertIsInstance(output.partition_metrics, unittest.mock.MagicMock)
        mock_to_metric_utility.assert_not_called()
        mock_create_for_public_partitions.assert_not_called()
        mock_create_for_private_partitions.assert_called_once_with(0.5)


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


def _get_partition_metrics(coef: int) -> metrics.PartitionMetrics:
    return metrics.PartitionMetrics(public_partitions=False,
                                    num_dataset_partitions=coef,
                                    num_non_public_partitions=2 * coef,
                                    num_empty_partitions=3 * coef,
                                    kept_partitions=metrics.MeanVariance(
                                        10 * coef, 11 * coef))


def _get_metric_utility(coef: int) -> metrics.MetricUtility:
    """Returns MetricUtility with numerical fields proportional to 'coef'"""
    get_mean_var = lambda: metrics.MeanVariance(coef, 2 * coef)
    get_bounding_errors = lambda: metrics.ContributionBoundingErrors(
        l0=get_mean_var(), linf_min=3 * coef, linf_max=4 * coef)
    get_value_errors = lambda: metrics.ValueErrors(
        bounding_errors=get_bounding_errors(),
        mean=5 * coef,
        variance=6 * coef,
        rmse=7 * coef,
        l1=8 * coef,
        rmse_with_dropped_partitions=9 * coef,
        l1_with_dropped_partitions=10 * coef)
    noise_std = 1000  # it's not merged, that's why not multiplied by coef.
    return metrics.MetricUtility(
        metric=pipeline_dp.Metrics.COUNT,
        noise_std=noise_std,
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        ratio_data_dropped=None,
        absolute_error=get_value_errors(),
        relative_error=get_value_errors().to_relative(10.0))


def _get_utility_report(coef: int) -> metrics.UtilityReport:
    return metrics.UtilityReport(
        input_aggregate_params=None,
        metric_errors=[
            _get_metric_utility(coef=coef),
            _get_metric_utility(coef=2 * coef)
        ],
        partition_metrics=_get_partition_metrics(coef=3 * coef))


class MergeMetricsTests(parameterized.TestCase):

    def test_merge_partition_selection_utilities(self):
        metrics1 = _get_partition_metrics(coef=1)
        metrics2 = _get_partition_metrics(coef=2)
        expected_utility = _get_partition_metrics(coef=3)
        cross_partition_combiners._merge_partition_metrics(metrics1, metrics2)
        self.assertEqual(metrics1, expected_utility)

    def test_merge_metric_utility(self):
        utility1 = _get_metric_utility(coef=2)
        utility2 = _get_metric_utility(coef=3)
        expected = _get_metric_utility(coef=5)
        cross_partition_combiners._merge_metric_utility(utility1, utility2)
        self.assertEqual(utility1, expected)

    def test_merge_utility_reports(self):
        report1 = _get_utility_report(coef=2)
        report2 = _get_utility_report(coef=5)
        expected_report = _get_utility_report(coef=7)
        cross_partition_combiners._merge_utility_reports(report1, report2)
        self.assertEqual(report1, expected_report)


class CrossPartitionCombiner(parameterized.TestCase):

    def _create_combiner(self, public_partitions=False):
        return cross_partition_combiners.CrossPartitionCombiner(
            dp_metrics=[pipeline_dp.Metrics.COUNT],
            public_partitions=public_partitions)

    def test_create_report_wo_mocks(self):
        combiner = self._create_combiner()
        per_partition_metrics = metrics.PerPartitionMetrics(
            0.2, metric_errors=[_get_sum_metrics()])
        utility_report = combiner.create_accumulator(per_partition_metrics)
        self.assertEqual(
            utility_report.partition_metrics.num_dataset_partitions, 1)
        self.assertLen(utility_report.metric_errors, 1)

    @patch("analysis.cross_partition_combiners._per_partition_to_utility_report"
          )
    def test_create_report_with_mocks(self,
                                      mock_per_partition_to_utility_report):
        combiner = self._create_combiner()
        per_partition_metrics = metrics.PerPartitionMetrics(
            0.2, metric_errors=[_get_sum_metrics()])
        combiner.create_accumulator(per_partition_metrics)
        expected_metrics = [pipeline_dp.Metrics.COUNT]
        expected_public_partitions = False
        mock_per_partition_to_utility_report.assert_called_once_with(
            per_partition_metrics, expected_metrics, expected_public_partitions)

    def test_create_accumulator(self):
        combiner = self._create_combiner()
        report1 = _get_utility_report(coef=2)
        report2 = _get_utility_report(coef=5)
        expected_report = _get_utility_report(coef=7)
        self.assertEqual(combiner.merge_accumulators(report1, report2),
                         expected_report)

    @parameterized.parameters(False, True)
    @patch(
        "analysis.cross_partition_combiners._multiply_float_dataclasses_field")
    def test_compute_metrics(self, public_partitions,
                             mock_multiply_float_dataclasses_field):
        combiner = self._create_combiner(public_partitions)
        report = _get_utility_report(coef=1)
        combiner.compute_metrics(report)
        expeced_num_output_partitions = 12 if public_partitions else 30
        mock_multiply_float_dataclasses_field.assert_called_once_with(
            report, 1.0 / expeced_num_output_partitions)


if __name__ == '__main__':
    absltest.main()
