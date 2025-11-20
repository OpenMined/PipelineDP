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
from unittest import mock
from unittest.mock import patch

from analysis import metrics
from analysis import cross_partition_combiners
from pipeline_dp import aggregate_params


def _get_default_sum_metrics(metric=aggregate_params.Metrics.COUNT, sum=10.0):
    return metrics.SumMetrics(aggregation=metric,
                              sum=sum,
                              clipping_to_min_error=0.0,
                              clipping_to_max_error=-5.0,
                              expected_l0_bounding_error=-2.0,
                              std_l0_bounding_error=3.0,
                              std_noise=4.0,
                              noise_kind=aggregate_params.NoiseKind.LAPLACE)


class PerPartitionToCrossPartitionMetrics(parameterized.TestCase):

    @parameterized.product(
        metric=[aggregate_params.Metrics.COUNT, aggregate_params.Metrics.SUM],
        keep_prob=[1, 0.25])
    def test_metric_utility(self, metric: aggregate_params.Metric,
                            keep_prob: float):
        input = metrics.SumMetrics(
            aggregation=metric,
            sum=10.0,
            clipping_to_min_error=0.0,
            clipping_to_max_error=-5.0,
            expected_l0_bounding_error=-2.0,
            std_l0_bounding_error=3.0,
            std_noise=4.0,
            noise_kind=aggregate_params.NoiseKind.LAPLACE)
        output: metrics.MetricUtility = cross_partition_combiners._sum_metrics_to_metric_utility(
            input,
            metric,
            partition_keep_probability=keep_prob,
            partition_weight=keep_prob)

        self.assertEqual(output.metric, metric)
        self.assertEqual(output.noise_kind, input.noise_kind)
        self.assertEqual(output.noise_std, input.std_noise)

        # Check absolute_error.
        abs_error: metrics.ValueErrors = output.absolute_error
        # (expected_l0_bounding_error + clipping_to_min_error +
        # clipping_to_max_error) = -7
        self.assertEqual(abs_error.mean, (-2 + 0 + (-5)) * keep_prob)
        # (std_l0_bounding_error**2 + std_noise**2) = 25
        self.assertEqual(abs_error.variance, (3**2 + 4**2) * keep_prob)
        # sqrt(mean ** 2 + variance)
        self.assertAlmostEqual(abs_error.rmse,
                               math.sqrt(7**2 + 25) * keep_prob,
                               delta=1e-12)

        bounding_errors = abs_error.bounding_errors
        self.assertEqual(
            bounding_errors.l0,
            metrics.MeanVariance(-2.0 * keep_prob, (3**2) * keep_prob))
        self.assertEqual(bounding_errors.linf_min, 0.0 * keep_prob)
        self.assertEqual(bounding_errors.linf_max, -5.0 * keep_prob)

        # Check relative_error.
        expected_rel_error = abs_error.to_relative(input.sum)
        self.assertEqual(output.relative_error, expected_rel_error)
        # mean / sum = -7 / 10
        self.assertAlmostEqual(output.relative_error.mean,
                               -0.7 * keep_prob,
                               delta=1e-12)

    @parameterized.parameters(False, True)
    def test_create_partition_metrics_for_public_partitions(
            self, is_empty_partition):
        output: metrics.PartitionsInfo = cross_partition_combiners._partition_metrics_public_partitions(
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
            partition_selection_probability_to_keep=0.2,
            raw_statistics=metrics.RawStatistics(privacy_id_count=10, count=15),
            metric_errors=[
                _get_default_sum_metrics(
                    metric=aggregate_params.Metrics.PRIVACY_ID_COUNT),
                _get_default_sum_metrics(metric=aggregate_params.Metrics.COUNT),
                _get_default_sum_metrics(metric=aggregate_params.Metrics.SUM)
            ])
        dp_metrics = [
            aggregate_params.Metrics.PRIVACY_ID_COUNT,
            aggregate_params.Metrics.COUNT, aggregate_params.Metrics.SUM
        ]
        cross_partition_combiners._per_partition_to_utility_report(
            per_partition_utility,
            dp_metrics,
            public_partitions,
            partition_weight=0.2)
        if public_partitions:
            mock_create_for_public_partitions.assert_called_once_with(False)
            mock_create_for_private_partitions.assert_not_called()
        else:
            mock_create_for_public_partitions.assert_not_called()
            mock_create_for_private_partitions.assert_called_once_with(0.2)

        self.assertEqual(mock_sum_metrics_to_metric_utility.call_count, 3)

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
        per_partition_utility = metrics.PerPartitionMetrics(
            partition_selection_probability_to_keep=0.5,
            raw_statistics=metrics.RawStatistics(privacy_id_count=3, count=100),
            metric_errors=None)
        output = cross_partition_combiners._per_partition_to_utility_report(
            per_partition_utility, [],
            public_partitions=False,
            partition_weight=0.5)

        self.assertIsNone(output.metric_errors)
        self.assertIsInstance(output.partitions_info, mock.MagicMock)
        mock_to_metric_utility.assert_not_called()
        mock_create_for_public_partitions.assert_not_called()
        mock_create_for_private_partitions.assert_called_once_with(0.5)

    def test_sum_metrics_to_data_dropped_count(self):
        input = metrics.SumMetrics(
            aggregation=aggregate_params.Metrics.COUNT,
            sum=10.0,
            clipping_to_min_error=0.0,
            clipping_to_max_error=-5.0,
            expected_l0_bounding_error=-2.0,
            std_l0_bounding_error=3.0,
            std_noise=4.0,
            noise_kind=aggregate_params.NoiseKind.LAPLACE)
        output = cross_partition_combiners._sum_metrics_to_data_dropped(
            input,
            partition_keep_probability=0.5,
            dp_metric=aggregate_params.Metrics.COUNT)
        self.assertEqual(
            output,
            metrics.DataDropInfo(l0=2.0, linf=5.0, partition_selection=1.5))

    def test_sum_metrics_to_data_dropped_sum(self):
        input = metrics.SumMetrics(
            aggregation=aggregate_params.Metrics.SUM,
            sum=12,
            clipping_to_min_error=3.0,
            clipping_to_max_error=-5.0,
            expected_l0_bounding_error=-2.0,
            std_l0_bounding_error=3.0,
            std_noise=4.0,
            noise_kind=aggregate_params.NoiseKind.LAPLACE)

        output = cross_partition_combiners._sum_metrics_to_data_dropped(
            input,
            partition_keep_probability=0.5,
            dp_metric=aggregate_params.Metrics.SUM)

        self.assertEqual(
            output,
            metrics.DataDropInfo(l0=2.0, linf=8.0, partition_selection=1.0))

    def test_sum_metrics_to_data_dropped_public_partition(self):
        input = _get_default_sum_metrics(metric=aggregate_params.Metrics.COUNT)
        output = cross_partition_combiners._sum_metrics_to_data_dropped(
            input,
            partition_keep_probability=1.0,
            dp_metric=aggregate_params.Metrics.COUNT)
        self.assertEqual(
            output,
            metrics.DataDropInfo(l0=2.0, linf=5.0, partition_selection=0.0))

    def test_sum_metrics_to_data_dropped_empty_public(self):
        input = metrics.SumMetrics(
            aggregation=aggregate_params.Metrics.SUM,
            sum=0.0,
            clipping_to_min_error=0.0,
            clipping_to_max_error=0.0,
            expected_l0_bounding_error=0.0,
            std_l0_bounding_error=0.0,
            std_noise=1.0,
            noise_kind=aggregate_params.NoiseKind.LAPLACE)
        output = cross_partition_combiners._sum_metrics_to_data_dropped(
            input,
            partition_keep_probability=1.0,
            dp_metric=aggregate_params.Metrics.COUNT)
        self.assertEqual(
            output,
            metrics.DataDropInfo(l0=0.0, linf=0.0, partition_selection=0.0))


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


def _get_partition_metrics(coef: int) -> metrics.PartitionsInfo:
    return metrics.PartitionsInfo(public_partitions=False,
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
        metric=aggregate_params.Metrics.COUNT,
        noise_std=noise_std,
        noise_kind=aggregate_params.NoiseKind.LAPLACE,
        ratio_data_dropped=None,
        absolute_error=get_value_errors(),
        relative_error=get_value_errors().to_relative(10.0))


def _get_utility_report(coef: int) -> metrics.UtilityReport:
    return metrics.UtilityReport(configuration_index=-1,
                                 metric_errors=[
                                     _get_metric_utility(coef=coef),
                                     _get_metric_utility(coef=2 * coef)
                                 ],
                                 partitions_info=_get_partition_metrics(coef=3 *
                                                                        coef))


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

    def _create_combiner(self,
                         dp_metrics=(aggregate_params.Metrics.COUNT,),
                         public_partitions=False,
                         weight_fn=cross_partition_combiners.equal_weight_fn):
        return cross_partition_combiners.CrossPartitionCombiner(
            dp_metrics, public_partitions, weight_fn)

    def test_create_report_wo_mocks(self):
        public_partitions = False
        prob_keep = 0.2
        combiner = self._create_combiner(
            public_partitions=public_partitions,
            weight_fn=cross_partition_combiners.equal_weight_fn)
        per_partition_metrics = metrics.PerPartitionMetrics(
            partition_selection_probability_to_keep=prob_keep,
            raw_statistics=metrics.RawStatistics(privacy_id_count=3, count=9),
            metric_errors=[_get_default_sum_metrics(sum=10.0)])
        sum_actual, utility_report, weight = combiner.create_accumulator(
            per_partition_metrics)
        self.assertEqual(sum_actual, (10.0,))
        self.assertEqual(utility_report.partitions_info.num_dataset_partitions,
                         1)
        self.assertLen(utility_report.metric_errors, 1)
        self.assertEqual(weight, prob_keep)

    def test_create_report_partition_size_is_used_as_weight_wo_mocks(self):
        combiner = self._create_combiner(
            weight_fn=cross_partition_combiners.partition_size_weight_fn)
        per_partition_metrics = metrics.PerPartitionMetrics(
            partition_selection_probability_to_keep=0.2,
            raw_statistics=metrics.RawStatistics(privacy_id_count=3, count=9),
            metric_errors=[_get_default_sum_metrics(sum=5.0)])
        _, _, weight = combiner.create_accumulator(per_partition_metrics)
        self.assertEqual(weight, 5.0)

    @patch(
        "analysis.cross_partition_combiners._per_partition_to_utility_report")
    def test_create_report_with_mocks(self,
                                      mock_per_partition_to_utility_report):
        dp_metrics = [aggregate_params.Metrics.COUNT]
        public_partitions = False
        prob_keep = 0.2
        combiner = self._create_combiner(
            dp_metrics,
            public_partitions,
            weight_fn=cross_partition_combiners.equal_weight_fn)
        per_partition_metrics = metrics.PerPartitionMetrics(
            partition_selection_probability_to_keep=prob_keep,
            raw_statistics=metrics.RawStatistics(privacy_id_count=3, count=9),
            metric_errors=[
                _get_default_sum_metrics(metric=aggregate_params.Metrics.COUNT)
            ])
        combiner.create_accumulator(per_partition_metrics)
        mock_per_partition_to_utility_report.assert_called_once_with(
            per_partition_metrics, dp_metrics, public_partitions, prob_keep)

    def test_create_accumulator(self):
        combiner = self._create_combiner()
        report1 = _get_utility_report(coef=2)
        acc1 = ((1,), report1, 0.5)
        report2 = _get_utility_report(coef=5)
        acc2 = ((3,), report2, 0.5)
        expected_report = _get_utility_report(coef=7)
        sum_actual, output_report, total_weight = combiner.merge_accumulators(
            acc1, acc2)
        self.assertEqual(sum_actual, (4,))
        self.assertEqual(output_report, expected_report)
        self.assertEqual(total_weight, 1)

    @patch("analysis.cross_partition_combiners._average_utility_report")
    def test_compute_metrics(self, mock_average_utility_report):
        combiner = self._create_combiner()
        report = _get_utility_report(coef=1)
        sum_actual_metrics = (1000,)
        # Actual value does not matter in the test.
        total_weight = 11
        acc = (sum_actual_metrics, report, total_weight)
        output = combiner.compute_metrics(acc)
        mock_average_utility_report.assert_called_once_with(
            output, sum_actual_metrics, total_weight)
        # Check that the input report was not modified.
        self.assertEqual(report, _get_utility_report(coef=1))


if __name__ == '__main__':
    absltest.main()
