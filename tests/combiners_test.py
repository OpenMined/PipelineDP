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
import typing
import unittest.mock as mock
from typing import List, Optional
from unittest.mock import patch

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
import pipeline_dp.budget_accounting as ba
import pipeline_dp.combiners as dp_combiners
from pipeline_dp import aggregate_params, NormKind, NoiseKind


class EmptyCombiner(dp_combiners.Combiner):
    """Empty combiner implementation for mocking."""

    def create_accumulator(self, values):
        return None

    def merge_accumulators(self, accumulator1, accumulator2):
        return None

    def compute_metrics(self, accumulator):
        return None

    def metrics_names(self) -> List[str]:
        return []

    def explain_computation(self):
        return None


def _create_mechanism_spec(
    no_noise: bool,
    mechanism_type: pipeline_dp.MechanismType = pipeline_dp.MechanismType.
    GAUSSIAN
) -> ba.MechanismSpec:
    if no_noise:
        eps, delta = 1e5, 1.0 - 1e-5
    else:
        eps, delta = 1, 1e-5

    return ba.MechanismSpec(mechanism_type, None, eps, delta)


def _create_aggregate_params(
        max_value: float = 1,
        max_partition_contributed: int = 2,
        vector_size: int = 1,
        vector_norm_kind: NormKind = pipeline_dp.NormKind.Linf,
        vector_max_norm: int = 5,
        output_noise_stddev: bool = False,
        noise_kind: pipeline_dp.NoiseKind = pipeline_dp.NoiseKind.GAUSSIAN):
    return pipeline_dp.AggregateParams(
        min_value=0,
        max_value=max_value,
        max_partitions_contributed=max_partition_contributed,
        max_contributions_per_partition=3,
        noise_kind=noise_kind,
        metrics=[pipeline_dp.Metrics.COUNT],
        vector_norm_kind=vector_norm_kind,
        vector_max_norm=vector_max_norm,
        vector_size=vector_size,
        output_noise_stddev=output_noise_stddev)


class CreateCompoundCombinersTest(parameterized.TestCase):

    def _create_aggregate_params(self, metrics: typing.Optional[typing.List]):
        return pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=metrics,
            min_value=0,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            budget_weight=10.0)

    @parameterized.named_parameters(
        dict(testcase_name='count',
             metrics=[pipeline_dp.Metrics.COUNT],
             expected_combiner_types=[dp_combiners.CountCombiner]),
        dict(testcase_name='sum',
             metrics=[pipeline_dp.Metrics.SUM],
             expected_combiner_types=[dp_combiners.SumCombiner]),
        dict(testcase_name='privacy_id_count',
             metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
             expected_combiner_types=[dp_combiners.PrivacyIdCountCombiner]),
        dict(testcase_name='count, sum, privacy_id_count',
             metrics=[
                 pipeline_dp.Metrics.SUM, pipeline_dp.Metrics.COUNT,
                 pipeline_dp.Metrics.PRIVACY_ID_COUNT
             ],
             expected_combiner_types=[
                 dp_combiners.CountCombiner, dp_combiners.SumCombiner,
                 dp_combiners.PrivacyIdCountCombiner
             ]),
        dict(testcase_name='mean',
             metrics=[
                 pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
                 pipeline_dp.Metrics.MEAN
             ],
             expected_combiner_types=[dp_combiners.MeanCombiner]),
        dict(testcase_name='variance',
             metrics=[
                 pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
                 pipeline_dp.Metrics.MEAN, pipeline_dp.Metrics.VARIANCE
             ],
             expected_combiner_types=[dp_combiners.VarianceCombiner]),
        dict(testcase_name='vector_sum',
             metrics=[pipeline_dp.Metrics.VECTOR_SUM],
             expected_combiner_types=[dp_combiners.VectorSumCombiner]),
        dict(testcase_name='percentiles',
             metrics=[
                 pipeline_dp.Metrics.PERCENTILE(10),
                 pipeline_dp.Metrics.PERCENTILE(90)
             ],
             expected_combiner_types=[dp_combiners.QuantileCombiner]),
    )
    def test_create_compound_combiner(self, metrics, expected_combiner_types):
        # Arrange.
        aggregate_params = self._create_aggregate_params(metrics)

        # Mock budget accountant.
        budget_accountant = mock.Mock()
        n_budget_requests = len(expected_combiner_types)
        if dp_combiners.MeanCombiner in expected_combiner_types:
            n_budget_requests += 1
        mock_budgets, expected_budgets = [], []
        for i, expeced_type in enumerate(expected_combiner_types):
            if expeced_type == dp_combiners.MeanCombiner:
                # MeanCombiner requests budgets twice, one for Count and one for
                # Sum.
                budgets = (f"budget{i}_count", f"budget{i}_sum")
                mock_budgets.extend(budgets)
                expected_budgets.append(budgets)
            else:
                budget = f"budget{i}"
                mock_budgets.append(budget)
                expected_budgets.append(budget)

        budget_accountant.request_budget = mock.Mock(side_effect=mock_budgets)

        # Act.
        compound_combiner = dp_combiners.create_compound_combiner(
            aggregate_params, budget_accountant)

        # Assert
        budget_accountant.request_budget.assert_called_with(
            pipeline_dp.aggregate_params.MechanismType.GAUSSIAN,
            weight=aggregate_params.budget_weight)
        # Check correctness of internal combiners
        combiners = compound_combiner._combiners
        self.assertLen(combiners, len(expected_combiner_types))
        for combiner, expect_type, expected_budget in zip(
                combiners, expected_combiner_types, expected_budgets):
            self.assertIsInstance(combiner, expect_type)
            self.assertEqual(combiner.mechanism_spec(), expected_budget)

    @patch.multiple("pipeline_dp.combiners.CustomCombiner",
                    __abstractmethods__=set())  # Mock CustomCombiner
    def test_create_compound_combiner_with_custom_combiners(self):
        # Arrange.
        # Create Mock CustomCombiners.
        custom_combiners = [
            dp_combiners.CustomCombiner(),
            dp_combiners.CustomCombiner()
        ]

        # Mock request budget and metrics names functions.
        for i, combiner in enumerate(custom_combiners):
            combiner.request_budget = mock.Mock()

        aggregate_params = self._create_aggregate_params(None)

        budget_accountant = pipeline_dp.NaiveBudgetAccountant(1, 1e-10)

        # Act
        compound_combiner = dp_combiners.create_compound_combiner_with_custom_combiners(
            aggregate_params, budget_accountant, custom_combiners)

        # Assert
        self.assertFalse(compound_combiner._return_named_tuple)
        for combiner in custom_combiners:
            combiner.request_budget.assert_called_once()

    def test_create_compound_combiner_with_post_aggregation(self):
        # Arrange.
        params = self._create_aggregate_params(
            [pipeline_dp.Metrics.PRIVACY_ID_COUNT])
        params.post_aggregation_thresholding = True
        params.budget_weight = 1

        # Mock budget accountant.
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(
            1.5, 1e-10, num_aggregations=1)

        # Act.
        compound_combiner = dp_combiners.create_compound_combiner(
            params, budget_accountant)
        budget_accountant._compute_budget_for_aggregation(params.budget_weight)
        budget_accountant.compute_budgets()

        # Assert
        # Check correctness of internal combiners
        combiners = compound_combiner._combiners
        self.assertLen(combiners, 1)
        self.assertIsInstance(combiners[0],
                              dp_combiners.PostAggregationThresholdingCombiner)
        mechanism_spec = combiners[0].mechanism_spec()
        self.assertEqual(mechanism_spec.mechanism_type,
                         pipeline_dp.MechanismType.GAUSSIAN_THRESHOLDING)
        self.assertEqual(mechanism_spec.eps, 1.5)
        self.assertEqual(mechanism_spec.delta, 1e-10)


class CountCombinerTest(parameterized.TestCase):

    def _create_combiner(
        self,
        no_noise: bool,
        mechanism_type: pipeline_dp.MechanismType = pipeline_dp.MechanismType.
        GAUSSIAN,
        output_noise_stddev: bool = False,
    ) -> dp_combiners.CountCombiner:
        mechanism_spec = _create_mechanism_spec(no_noise, mechanism_type)
        aggregate_params = _create_aggregate_params(
            output_noise_stddev=output_noise_stddev)
        return dp_combiners.CountCombiner(mechanism_spec, aggregate_params)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(2, combiner.create_accumulator([1, 2]))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.merge_accumulators(0, 0))
        self.assertEqual(5, combiner.merge_accumulators(1, 4))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        output = combiner.compute_metrics(3)
        self.assertLen(output, 1)
        self.assertAlmostEqual(3, output['count'], delta=1e-5)

    @patch("pipeline_dp.dp_computations.GaussianMechanism.add_noise")
    def test_compute_metrics_with_noise(self, mock_add_noise):
        combiner = self._create_combiner(no_noise=False)
        accumulator = 500
        noised_value = 510
        mock_add_noise.return_value = noised_value
        noisy_value = combiner.compute_metrics(accumulator)['count']
        self.assertEqual(noisy_value, noised_value)
        mock_add_noise.assert_called_once_with(accumulator)

    @parameterized.named_parameters(
        dict(testcase_name='gaussian',
             mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
             expected_noise_param=15.835),
        dict(testcase_name='laplace',
             mechanism_type=pipeline_dp.MechanismType.LAPLACE,
             expected_noise_param=6.0),
    )
    def test_mechanism(self, mechanism_type: pipeline_dp.MechanismType,
                       expected_noise_param: float):
        combiner = self._create_combiner(no_noise=False,
                                         mechanism_type=mechanism_type)
        mechanism = combiner.get_mechanism()
        self.assertEqual(mechanism.noise_kind, mechanism_type.to_noise_kind())
        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_param,
                               delta=1e-3)

    def test_explain_computation(self):
        combiner = self._create_combiner(no_noise=False)
        expected = ("Computed DP count.*\n.*Gaussian mechanism:  "
                    "parameter=15.8.*eps=1.0  delta=1e-05.*l2_sensitivity=4.2")
        self.assertRegex(combiner.explain_computation()(), expected)

    def test_noise_stddev(self):
        combiner = self._create_combiner(
            no_noise=False,
            mechanism_type=pipeline_dp.MechanismType.LAPLACE,
            output_noise_stddev=True)
        output = combiner.compute_metrics(5)
        self.assertLen(output, 2)
        # For COUNT and Laplace noise
        # stddev = 1/eps*max_partitions_contributed*max_contributions_per_partition*sqrt(2)
        expected_stddev = 1 / 1 * 2 * 3 * np.sqrt(2)
        self.assertAlmostEqual(output['count_noise_stddev'],
                               expected_stddev,
                               delta=1e-8)
        # check that noised count is within 10 stddev, which for Laplace
        # should not be with probability 7.213541e-07 (=flakiness probability)
        self.assertTrue(2 - 10 * expected_stddev <= output["count"] <= 2 +
                        10 * expected_stddev)


class PrivacyIdCountCombinerTest(parameterized.TestCase):

    def _create_combiner(
        self,
        no_noise: bool,
        mechanism_type: pipeline_dp.MechanismType = pipeline_dp.MechanismType.
        GAUSSIAN,
        output_noise_stddev: bool = False,
    ) -> dp_combiners.PrivacyIdCountCombiner:
        mechanism_spec = _create_mechanism_spec(no_noise, mechanism_type)
        aggregate_params = _create_aggregate_params(
            output_noise_stddev=output_noise_stddev)
        return dp_combiners.PrivacyIdCountCombiner(mechanism_spec,
                                                   aggregate_params)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(1, combiner.create_accumulator([1, 2]))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual(0, combiner.merge_accumulators(0, 0))
        self.assertEqual(5, combiner.merge_accumulators(1, 4))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        self.assertAlmostEqual(3,
                               combiner.compute_metrics(3)['privacy_id_count'],
                               delta=1e-5)

    @patch("pipeline_dp.dp_computations.GaussianMechanism.add_noise")
    def test_compute_metrics_with_noise(self, mock_add_noise):
        combiner = self._create_combiner(no_noise=False)
        accumulator = 500
        noised_value = 510
        mock_add_noise.return_value = noised_value
        noisy_value = combiner.compute_metrics(accumulator)['privacy_id_count']
        self.assertEqual(noisy_value, noised_value)
        mock_add_noise.assert_called_once_with(accumulator)

    @parameterized.named_parameters(
        dict(testcase_name='gaussian',
             mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
             expected_noise_param=5.278),
        dict(testcase_name='laplace',
             mechanism_type=pipeline_dp.MechanismType.LAPLACE,
             expected_noise_param=2.0),
    )
    def test_mechanism(self, mechanism_type: pipeline_dp.MechanismType,
                       expected_noise_param: float):
        combiner = self._create_combiner(no_noise=False,
                                         mechanism_type=mechanism_type)
        mechanism = combiner.get_mechanism()
        self.assertEqual(mechanism.noise_kind, mechanism_type.to_noise_kind())
        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_param,
                               delta=1e-3)

    def test_explain_computation(self):
        combiner = self._create_combiner(no_noise=False)
        expected = ("Computed DP privacy_id_count.*\n.*Gaussian mechanism:  "
                    "parameter=5.2.*eps=1.0  delta=1e-05.*l2_sensitivity=1.4")
        self.assertRegex(combiner.explain_computation()(), expected)

    def test_noise_stddev(self):
        combiner = self._create_combiner(
            no_noise=False,
            mechanism_type=pipeline_dp.MechanismType.LAPLACE,
            output_noise_stddev=True)
        output = combiner.compute_metrics(5)
        self.assertLen(output, 2)
        # For PRIVACY_ID_COUNT and Laplace stddev = 1/eps*max_partitions_contributed*sqrt(2)
        expected_stddev = 1 / 1 * 2 * np.sqrt(2)
        self.assertAlmostEqual(output['privacy_id_count_noise_stddev'],
                               expected_stddev,
                               delta=1e-8)
        # check that noised count is within 10 stddev, which for Laplace
        # should be with probability 7.213541e-07 (=flakiness probability)
        self.assertTrue(
            2 - 10 * expected_stddev <= output["privacy_id_count"] <= 2 +
            10 * expected_stddev)


class PostAggregationThresholdingCombinerTest(parameterized.TestCase):

    def _create_combiner(
        self,
        small_noise: bool = False,
        noise_kind: pipeline_dp.NoiseKind = pipeline_dp.NoiseKind.GAUSSIAN,
        pre_threshold: Optional[int] = None
    ) -> dp_combiners.PostAggregationThresholdingCombiner:
        eps, delta = (10**3, 0.1) if small_noise else (1, 1e-10)
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(eps, delta)
        aggregate_params = _create_aggregate_params()
        aggregate_params.noise_kind = noise_kind
        aggregate_params.pre_threshold = pre_threshold
        combiner = dp_combiners.PostAggregationThresholdingCombiner(
            budget_accountant, aggregate_params)
        budget_accountant.compute_budgets()
        return combiner

    def _get_mechanism_type(self, noise_kind: pipeline_dp.NoiseKind):
        if noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
            return aggregate_params.MechanismType.GAUSSIAN_THRESHOLDING
        if noise_kind == pipeline_dp.NoiseKind.LAPLACE:
            return aggregate_params.MechanismType.LAPLACE_THRESHOLDING

    def _get_strategy(self, noise_kind: pipeline_dp.NoiseKind):
        if noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
            return pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING
        if noise_kind == pipeline_dp.NoiseKind.LAPLACE:
            return pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING

    @parameterized.named_parameters(
        dict(testcase_name='gaussian',
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             expected_strategy_class="GaussianPartitionSelectionStrategy"),
        dict(testcase_name='laplace',
             noise_kind=pipeline_dp.NoiseKind.LAPLACE,
             expected_strategy_class="LaplacePartitionSelectionStrategy"),
    )
    def test_create_combiner(self, noise_kind: pipeline_dp.NoiseKind,
                             expected_strategy_class: str):
        # Arrange/act
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-10)
        aggregate_params = _create_aggregate_params()
        aggregate_params.noise_kind = noise_kind
        combiner = dp_combiners.PostAggregationThresholdingCombiner(
            budget_accountant, aggregate_params)
        budget_accountant.compute_budgets()

        # Assert
        expected_mechanism_type = self._get_mechanism_type(noise_kind)
        self.assertEqual(
            combiner.mechanism_spec(),
            ba.MechanismSpec(expected_mechanism_type, None, 1, 1e-10))
        self.assertEqual(combiner.sensitivities().l0, 2)
        self.assertEqual(combiner.sensitivities().linf, 1)
        thresholding_strategy = combiner.create_mechanism(
        )._thresholding_strategy
        self.assertEqual(
            type(thresholding_strategy).__name__, expected_strategy_class)

    def test_create_accumulator(self):
        combiner = self._create_combiner()
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(1, combiner.create_accumulator([1, 2]))

    def test_merge_accumulators(self):
        combiner = self._create_combiner()
        self.assertEqual(0, combiner.merge_accumulators(0, 0))
        self.assertEqual(5, combiner.merge_accumulators(1, 4))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(small_noise=True)
        self.assertAlmostEqual(3,
                               combiner.compute_metrics(3)['privacy_id_count'],
                               delta=1e-2)

    @patch(
        'pydp._pydp._partition_selection.GaussianPartitionSelectionStrategy.noised_value_if_should_keep'
    )
    def test_noised_value_if_should_keep(self, mock_function):
        combiner = self._create_combiner(False)
        mock_function.return_value = "output"
        self.assertEqual(
            combiner.compute_metrics(100)['privacy_id_count'], "output")
        mock_function.assert_called_once_with(100)

    @parameterized.named_parameters(
        dict(testcase_name='gaussian',
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             pre_threshold=None),
        dict(testcase_name='laplace',
             noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
             pre_threshold=20),
    )
    @patch(
        'pipeline_dp.partition_selection.create_partition_selection_strategy')
    def test_mechanism(self, mock_create_partition_selection_strategy,
                       noise_kind: pipeline_dp.NoiseKind,
                       pre_threshold: Optional[int]):
        combiner = self._create_combiner(False, noise_kind, pre_threshold)
        combiner.get_mechanism()

        expected_strategy = self._get_strategy(noise_kind)
        mock_create_partition_selection_strategy.assert_called_once_with(
            expected_strategy, 1.0, 1e-10, 2, pre_threshold)

    def test_explain_computation(self):
        combiner = self._create_combiner()
        expected = ('Computed DP privacy_id_count with\n     Gaussian '
                    'Thresholding with threshold=56.5 eps=1.0 delta=1e-10')
        self.assertRegex(combiner.explain_computation()(), expected)


class SumCombinerTest(parameterized.TestCase):

    def _create_aggregate_params_per_partition_bound(
            self, output_noise_stddev: bool = False):
        return pipeline_dp.AggregateParams(
            min_sum_per_partition=0,
            max_sum_per_partition=3,
            max_contributions_per_partition=1,
            max_partitions_contributed=1,
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.SUM],
            output_noise_stddev=output_noise_stddev)

    def _create_combiner(
        self,
        no_noise: bool,
        per_partition_bound: bool,
        max_value=1.0,
        mechanism_type: pipeline_dp.MechanismType = pipeline_dp.MechanismType.
        GAUSSIAN,
        output_noise_stddev: bool = False,
    ) -> dp_combiners.SumCombiner:
        mechanism_spec = _create_mechanism_spec(no_noise, mechanism_type)
        if per_partition_bound:
            aggr_params = self._create_aggregate_params_per_partition_bound()
        else:
            aggr_params = _create_aggregate_params(
                max_value=max_value, output_noise_stddev=output_noise_stddev)
            aggr_params.metrics = [pipeline_dp.Metrics.SUM]
        return dp_combiners.SumCombiner(mechanism_spec, aggr_params)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator_per_contribution_bounding(self, no_noise):
        combiner = self._create_combiner(no_noise, per_partition_bound=False)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(2, combiner.create_accumulator([1, 1]))
        # Bounding on values.
        self.assertEqual(2, combiner.create_accumulator([1, 3]))
        self.assertEqual(1, combiner.create_accumulator([0, 3]))
        self.assertTrue(combiner.expects_per_partition_sampling())

    def test_create_accumulator_per_partition_bound(self):
        combiner = self._create_combiner(no_noise=False,
                                         per_partition_bound=True)
        self.assertEqual(0, combiner.create_accumulator([]))
        self.assertEqual(2.5, combiner.create_accumulator([2, 0.5]))
        # Clipping sum to [0, 3].
        self.assertEqual(3, combiner.create_accumulator([4, 1]))
        self.assertEqual(0, combiner.create_accumulator([-10, 5, 3]))
        self.assertFalse(combiner.expects_per_partition_sampling())

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True, per_partition_bound=True),
        dict(testcase_name='noise', no_noise=False, per_partition_bound=False),
    )
    def test_merge_accumulators(self, no_noise, per_partition_bound):
        combiner = self._create_combiner(no_noise, per_partition_bound)
        self.assertEqual(0, combiner.merge_accumulators(0, 0))
        self.assertEqual(5, combiner.merge_accumulators(1, 4))

    @parameterized.named_parameters(
        dict(testcase_name='per_contribution_bound', per_partition_bound=True),
        dict(testcase_name='per_partition_bound', per_partition_bound=False),
    )
    def test_compute_metrics_no_noise(self, per_partition_bound):
        combiner = self._create_combiner(
            no_noise=True, per_partition_bound=per_partition_bound)
        self.assertAlmostEqual(3,
                               combiner.compute_metrics(3)['sum'],
                               delta=1e-5)

    @patch("pipeline_dp.dp_computations.GaussianMechanism.add_noise")
    def test_compute_metrics_with_noise(self, mock_add_noise):
        combiner = self._create_combiner(no_noise=False,
                                         per_partition_bound=False)
        accumulator = 500
        noised_value = 510
        mock_add_noise.return_value = noised_value
        noisy_value = combiner.compute_metrics(accumulator)['sum']
        self.assertEqual(noisy_value, noised_value)
        mock_add_noise.assert_called_once_with(accumulator)

    @parameterized.named_parameters(
        dict(testcase_name='gaussian_per_partition_bounding',
             mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
             per_partition_bound=True,
             expected_noise_param=11.197),
        dict(testcase_name='gaussian',
             mechanism_type=pipeline_dp.MechanismType.GAUSSIAN,
             per_partition_bound=False,
             expected_noise_param=110.847),
        dict(testcase_name='laplace_per_partition_bounding',
             mechanism_type=pipeline_dp.MechanismType.LAPLACE,
             per_partition_bound=True,
             expected_noise_param=3.0),
        dict(testcase_name='laplace',
             mechanism_type=pipeline_dp.MechanismType.LAPLACE,
             per_partition_bound=False,
             expected_noise_param=42.0),
    )
    def test_mechanism(self, mechanism_type: pipeline_dp.MechanismType,
                       per_partition_bound: bool, expected_noise_param: float):
        combiner = self._create_combiner(
            no_noise=False,
            max_value=7.0,
            per_partition_bound=per_partition_bound,
            mechanism_type=mechanism_type)
        mechanism = combiner.get_mechanism()
        self.assertEqual(mechanism.noise_kind, mechanism_type.to_noise_kind())
        self.assertAlmostEqual(mechanism.noise_parameter,
                               expected_noise_param,
                               delta=1e-3)

    def test_explain_computation(self):
        combiner = self._create_combiner(no_noise=False,
                                         per_partition_bound=False)
        expected = ("Computed DP sum.*\n.*Gaussian mechanism:  "
                    "parameter=15.*eps=1.0  delta=1e-05.*l2_sensitivity=4.2")
        self.assertRegex(combiner.explain_computation()(), expected)

    def test_noise_stddev(self):
        combiner = self._create_combiner(
            no_noise=False,
            per_partition_bound=False,
            max_value=5,
            mechanism_type=pipeline_dp.MechanismType.LAPLACE,
            output_noise_stddev=True)
        output = combiner.compute_metrics(5)
        self.assertLen(output, 2)
        # For SUM and Laplace stddev = 1/eps*max_partitions_contributed*max_contributions_per_partition*max_value*sqrt(2)
        expected_stddev = 1 / 1 * 2 * 3 * 5 * np.sqrt(2)
        self.assertAlmostEqual(output['sum_noise_stddev'],
                               expected_stddev,
                               delta=1e-8)
        # check that noised count is within 10 stddev, which for Laplace
        # should be with probability 7.213541e-07 (=flakiness probability)
        self.assertTrue(2 - 10 * expected_stddev <= output["sum"] <= 2 +
                        10 * expected_stddev)


class MeanCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params(max_value=4)
        metrics_to_compute = ['count', 'sum', 'mean']
        return dp_combiners.MeanCombiner(mechanism_spec, mechanism_spec,
                                         aggregate_params, metrics_to_compute)

    def test_create_accumulator(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0), combiner.create_accumulator([]))
            self.assertEqual((2, 0), combiner.create_accumulator([1, 3]))

    def test_merge_accumulators(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0), combiner.merge_accumulators((0, 0),
                                                                 (0, 0)))
            self.assertEqual((5, 2), combiner.merge_accumulators((1, 0),
                                                                 (4, 2)))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        res = combiner.compute_metrics((3, 3))
        self.assertAlmostEqual(3, res['count'], delta=1e-5)
        self.assertAlmostEqual(9, res['sum'], delta=1e-5)
        self.assertAlmostEqual(3, res['mean'], delta=1e-5)

    @patch("pipeline_dp.dp_computations.GaussianMechanism.add_noise")
    def test_compute_metrics_with_noise(self, mock_add_noise):
        combiner = self._create_combiner(no_noise=False)
        count, normalized_sum = 100, 150

        # Set mock noise to be a deterministic addition of 1, which can be
        # easily tested.
        mock_add_noise.side_effect = lambda x: x + 1

        output = combiner.compute_metrics((count, normalized_sum))
        self.assertEqual(output["count"], 101)  # add_noise(100)
        # expected_sum = add_noise(normalized_sum) + mid_range*noisy_count = 353
        # More details are in dp_computations.MeanMechanism docstring.
        self.assertEqual(output["sum"], 353)
        self.assertAlmostEqual(output["mean"], 353 / 101, delta=1e-12)


class VarianceCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params(max_value=4)
        metrics_to_compute = ['count', 'sum', 'mean', 'variance']
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.VarianceCombiner(params, metrics_to_compute)

    def test_create_accumulator(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0, 0), combiner.create_accumulator([]))
            self.assertEqual((2, -1, 1), combiner.create_accumulator([1, 2]))

    def test_merge_accumulators(self):
        for no_noise in [False, True]:
            combiner = self._create_combiner(no_noise)
            self.assertEqual((0, 0, 0),
                             combiner.merge_accumulators((0, 0, 0), (0, 0, 0)))
            self.assertEqual((5, 2, 2),
                             combiner.merge_accumulators((1, 0, 0), (4, 2, 2)))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        # potential values: 1, 2, 2, 3 | middle = 2
        res = combiner.compute_metrics((4, 0, 2))
        self.assertAlmostEqual(4, res['count'], delta=1e-5)
        self.assertAlmostEqual(8, res['sum'], delta=1e-5)
        self.assertAlmostEqual(2, res['mean'], delta=1e-5)
        self.assertAlmostEqual(0.5, res['variance'], delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        # potential values: 1, 1, 2, 3, 3 | middle = 2
        count = 500
        sum_ = 1000
        normalized_sum = 0
        normalized_sum_of_squares = 4
        mean = 2
        variance = 0.8
        noisy_values = [
            combiner.compute_metrics(
                (count, normalized_sum, normalized_sum_of_squares))
            for _ in range(1000)
        ]

        noisy_counts = [noisy_value['count'] for noisy_value in noisy_values]
        self.assertAlmostEqual(count, np.mean(noisy_counts), delta=5)
        self.assertGreater(np.var(noisy_counts), 1)  # check that noise is added

        noisy_sums = [noisy_value['sum'] for noisy_value in noisy_values]
        self.assertAlmostEqual(sum_, np.mean(noisy_sums), delta=20)
        self.assertGreater(np.var(noisy_sums), 1)  # check that noise is added

        noisy_means = [noisy_value['mean'] for noisy_value in noisy_values]
        self.assertAlmostEqual(mean, np.mean(noisy_means), delta=5e-1)
        self.assertGreater(np.var(noisy_means),
                           0.01)  # check that noise is added

        noisy_variances = [
            noisy_value['variance'] for noisy_value in noisy_values
        ]
        self.assertAlmostEqual(variance, np.mean(noisy_variances), delta=20)
        self.assertGreater(np.var(noisy_variances),
                           0.01)  # check that noise is added


class QuantileCombinerTest(parameterized.TestCase):

    def _create_combiner(self,
                         no_noise: bool,
                         percentiles: typing.List[int] = [10, 90]):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params(max_value=1000)
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.QuantileCombiner(params,
                                             percentiles_to_compute=percentiles)

    def test_create_accumulator(self):
        combiner = self._create_combiner(no_noise=False)
        quantile_tree = combiner._create_empty_quantile_tree()
        self.assertEqual(16, quantile_tree.branching_factor)  # default value
        self.assertEqual(4, quantile_tree.height)  # default value

    def test_compute_metrics_without_merge(self):
        # Arrange.
        combiner = self._create_combiner(no_noise=True,
                                         percentiles=[10, 50, 90])

        # Act.
        # Add values 0, ... 1000.
        acc = combiner.create_accumulator(list(range(1001)))
        metrics = combiner.compute_metrics(acc)

        # Assert.
        self.assertLen(metrics, 3)

        # The budget is high, so computed percentiles should be close to actual.
        self.assertAlmostEqual(100, metrics['percentile_10'], delta=1e-1)
        self.assertAlmostEqual(500, metrics['percentile_50'], delta=1e-1)
        self.assertAlmostEqual(900, metrics['percentile_90'], delta=1e-1)

    def test_compute_metrics_with_merge(self):
        # Arrange.
        combiner = self._create_combiner(no_noise=True,
                                         percentiles=[10, 50, 90])

        # Act.
        # Add values 0, ... 1000, each value for a separate acc and merge them.
        result_acc = None
        for i in range(1001):
            acc = combiner.create_accumulator([i])
            result_acc = acc if i == 0 else combiner.merge_accumulators(
                acc, result_acc)

        metrics = combiner.compute_metrics(result_acc)

        # Assert.
        self.assertLen(metrics, 3)

        # The budget is high, so computed percentiles should be close to actual.
        self.assertAlmostEqual(100, metrics['percentile_10'], delta=1e-1)
        self.assertAlmostEqual(500, metrics['percentile_50'], delta=1e-1)
        self.assertAlmostEqual(900, metrics['percentile_90'], delta=1e-1)


class CompoundCombinerTest(parameterized.TestCase):

    def _create_combiner(self, no_noise):
        mechanism_spec = _create_mechanism_spec(no_noise)
        aggregate_params = _create_aggregate_params()
        return dp_combiners.CompoundCombiner([
            dp_combiners.CountCombiner(mechanism_spec, aggregate_params),
            dp_combiners.SumCombiner(mechanism_spec, aggregate_params)
        ],
                                             return_named_tuple=True)

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_create_accumulator(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual((1, (2, 2)), combiner.create_accumulator((1, 1)))
        self.assertEqual((1, (2, 2)), combiner.create_accumulator((1, 1)))
        self.assertEqual((1, (3, 2)), combiner.create_accumulator((0, 3, 4)))

    @parameterized.named_parameters(
        dict(testcase_name='no_noise', no_noise=True),
        dict(testcase_name='noise', no_noise=False),
    )
    def test_merge_accumulators(self, no_noise):
        combiner = self._create_combiner(no_noise)
        self.assertEqual((2, (4, 4)),
                         combiner.merge_accumulators((1, (2, 2)), (1, (2, 2))))
        self.assertEqual((3, (4, 5)),
                         combiner.merge_accumulators((2, (2, 3)), (1, (2, 2))))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True)
        metrics_tuple = combiner.compute_metrics((3, [2, 3]))
        self.assertAlmostEqual(2, metrics_tuple.count, delta=1e-5)
        self.assertAlmostEqual(3, metrics_tuple.sum, delta=1e-5)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False)
        accumulator = (2, (2, 3))
        noisy_values = [
            combiner.compute_metrics(accumulator) for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        noised_count = []
        noised_sum = []
        for metrics_tuple in noisy_values:
            noised_count.append(metrics_tuple.count)
            noised_sum.append(metrics_tuple.sum)

        self.assertAlmostEqual(accumulator[1][0],
                               np.mean(noised_count),
                               delta=2)
        self.assertAlmostEqual(accumulator[1][1], np.mean(noised_sum), delta=2)
        self.assertTrue(np.var(noised_count) > 1)  # check that noise is added
        self.assertTrue(np.var(noised_sum) > 1)  # check that noise is added

    def test_expects_per_partition_sampling(self):

        class MockCombiner(EmptyCombiner):

            def __init__(self, return_value: bool):
                self._return_value = return_value

            def expects_per_partition_sampling(self) -> bool:
                return self._return_value

        def create_combiner(return_values: List[bool]):
            combiners = [MockCombiner(v) for v in return_values]
            return dp_combiners.CompoundCombiner(combiners,
                                                 return_named_tuple=True)

        self.assertTrue(
            create_combiner([True]).expects_per_partition_sampling())
        self.assertTrue(
            create_combiner([True, False]).expects_per_partition_sampling())
        self.assertFalse(
            create_combiner([False, False]).expects_per_partition_sampling())


class VectorSumCombinerTest(parameterized.TestCase):

    def _create_combiner(
        self,
        no_noise: bool,
        vector_size: int,
        vector_norm_kind: NormKind = NormKind.Linf,
        vector_max_norm: int = 5,
        noise_kind: pipeline_dp.NoiseKind = pipeline_dp.NoiseKind.GAUSSIAN,
        max_partition_contributed: int = 2,
    ) -> dp_combiners.VectorSumCombiner:
        mechanism_type = pipeline_dp.MechanismType.GAUSSIAN
        if noise_kind == NoiseKind.LAPLACE:
            mechanism_type = pipeline_dp.MechanismType.LAPLACE
        mechanism_spec = _create_mechanism_spec(no_noise, mechanism_type)
        aggregate_params = _create_aggregate_params(
            vector_size=vector_size,
            max_partition_contributed=max_partition_contributed,
            vector_norm_kind=vector_norm_kind,
            vector_max_norm=vector_max_norm,
            noise_kind=noise_kind)
        params = dp_combiners.CombinerParams(mechanism_spec, aggregate_params)
        return dp_combiners.VectorSumCombiner(params)

    @parameterized.product(testcase=[
        dict(input_vector=[[]], output_vector=[]),
        dict(input_vector=[[0]], output_vector=[0]),
        dict(input_vector=[[0, 0]], output_vector=[0, 0]),
        dict(input_vector=[[1], [2]], output_vector=[3]),
        dict(input_vector=[[1, 2], [3, 4]], output_vector=[4, 6]),
        dict(input_vector=[[1, 2, 3], [4, 5, 6]], output_vector=[5, 7, 9])
    ],
                           norm_kind=[NormKind.Linf, NormKind.L1, NormKind.L2])
    def test_create_accumulator_sums_correctly(self, testcase, norm_kind):
        combiner = self._create_combiner(no_noise=True,
                                         vector_size=len(
                                             testcase['input_vector'][0]),
                                         vector_norm_kind=norm_kind,
                                         vector_max_norm=999)  # no clipping

        result = combiner.create_accumulator(testcase['input_vector'])

        np.testing.assert_array_equal(result, testcase['output_vector'])

    @parameterized.parameters(
        dict(norm_kind=NormKind.Linf,
             norm=5,
             input_vector=[[6]],
             output_vector=[5]),
        dict(norm_kind=NormKind.Linf,
             norm=5,
             input_vector=[[6], [7]],
             output_vector=[5]),
        dict(norm_kind=NormKind.Linf,
             norm=5,
             input_vector=[[5, 6]],
             output_vector=[5, 5]),
        dict(norm_kind=NormKind.L1,
             norm=10,
             input_vector=[[11]],
             output_vector=[10]),
        dict(norm_kind=NormKind.L1,
             norm=10,
             input_vector=[[5], [6]],
             output_vector=[10]),
        dict(norm_kind=NormKind.L1,
             norm=5,
             input_vector=[[6, 6]],
             output_vector=[2.5, 2.5]),
        dict(norm_kind=NormKind.L2,
             norm=5,
             input_vector=[[6]],
             output_vector=[5]),
        dict(norm_kind=NormKind.L2,
             norm=5,
             input_vector=[[4], [4]],
             output_vector=[5]),
        dict(norm_kind=NormKind.L2,
             norm=4,
             input_vector=[[3, 4]],
             output_vector=[2.4, 3.2]),
    )
    def test_create_accumulator_clips_correctly(self, norm_kind, norm,
                                                input_vector, output_vector):
        combiner = self._create_combiner(no_noise=True,
                                         vector_size=len(input_vector[0]),
                                         vector_norm_kind=norm_kind,
                                         vector_max_norm=norm)  # clips by norm

        result = combiner.create_accumulator(input_vector)

        np.testing.assert_almost_equal(result, output_vector, decimal=3)

    def test_merge_accumulators(self):
        combiner = self._create_combiner(no_noise=True, vector_size=1)
        self.assertEqual(
            np.array([0.]),
            combiner.merge_accumulators(np.array([0.]), np.array([0.])))
        combiner = self._create_combiner(no_noise=True, vector_size=2)
        merge_result = combiner.merge_accumulators(np.array([1., 1.]),
                                                   np.array([1., 4.]))
        self.assertTrue(np.array_equal(np.array([2., 5.]), merge_result))

    def test_compute_metrics_no_noise(self):
        combiner = self._create_combiner(no_noise=True, vector_size=1)
        self.assertAlmostEqual(5,
                               combiner.compute_metrics(np.array(
                                   [5]))['vector_sum'],
                               delta=1e-5)
    @parameterized.parameters(
        dict(noise_kind=NoiseKind.GAUSSIAN,
             norm_kind=NormKind.Linf),
        dict(noise_kind=NoiseKind.GAUSSIAN,
             norm_kind=NormKind.L2),
        dict(noise_kind=NoiseKind.LAPLACE,
             norm_kind=NormKind.Linf),
        dict(noise_kind=NoiseKind.LAPLACE,
             norm_kind=NormKind.L1),
    )
    def test_vector_sensitivity_not_per_component(self, noise_kind, norm_kind):
        # This tests checks that the noise added is close to zero.
        # If the noise is computed per component of the vector, the noise added would be much larger.
        vector_size = 10000
        combiner = self._create_combiner(no_noise=True,
                                         vector_size=vector_size,
                                         noise_kind=noise_kind,
                                         vector_norm_kind=norm_kind,
                                         vector_max_norm=1,
                                         max_partition_contributed=1)

        m = combiner.compute_metrics(np.array([0] * vector_size))

        for value in m['vector_sum']:
            self.assertAlmostEqual(value, 0, delta=1)

    def test_compute_metrics_with_noise(self):
        combiner = self._create_combiner(no_noise=False, vector_size=2)
        accumulator = np.array([1, 3])
        noisy_values = [
            combiner.compute_metrics(accumulator)['vector_sum']
            for _ in range(1000)
        ]
        # Standard deviation for the noise is about 1.37. So we set a large
        # delta here.
        mean_array = np.mean(noisy_values, axis=0)
        self.assertAlmostEqual(accumulator[0], mean_array[0], delta=5)
        self.assertAlmostEqual(accumulator[1], mean_array[1], delta=5)
        self.assertTrue(np.var(noisy_values) > 1)  # check that noise is added


if __name__ == '__main__':
    absltest.main()
