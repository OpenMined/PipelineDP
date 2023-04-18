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
from absl.testing import parameterized

import pipeline_dp
from pipeline_dp import aggregate_params
from pipeline_dp import budget_accounting
"""Aggregate Params Test"""

Metric = aggregate_params.Metric


class MetricsTest(parameterized.TestCase):

    def test_eq(self):
        self.assertEqual(Metric("name"), Metric("name"))
        self.assertEqual(Metric("name", 1), Metric("name", 1))
        self.assertNotEqual(Metric("name"), Metric("other_name"))
        self.assertNotEqual(Metric("name", 1), Metric("name", 2))
        self.assertNotEqual(Metric("name"), "name")  # different type

    def test_str(self):
        self.assertEqual(str(Metric("name")), "name")
        self.assertEqual(str(Metric("name", 10)), "name(10)")

    def test_hash(self):
        self.assertEqual(hash(Metric("name")), hash("name"))
        self.assertEqual(hash(Metric("name", 10)), hash("name(10)"))


class AggregateParamsTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='negative max_partitions_contributed',
             error_msg='max_partitions_contributed has to be positive integer',
             min_value=None,
             max_value=None,
             max_partitions_contributed=-1,
             max_contributions_per_partition=1,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
        dict(testcase_name='negative max_contributions_per_partition',
             error_msg=
             'max_contributions_per_partition has to be positive integer',
             min_value=None,
             max_value=None,
             max_partitions_contributed=1,
             max_contributions_per_partition=-1,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
        dict(testcase_name='float max_partitions_contributed',
             error_msg='max_partitions_contributed has to be positive integer',
             min_value=None,
             max_value=None,
             max_partitions_contributed=1.5,
             max_contributions_per_partition=1,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
        dict(testcase_name='float max_contributions_per_partition',
             error_msg=
             'max_contributions_per_partition has to be positive integer',
             min_value=None,
             max_value=None,
             max_partitions_contributed=1,
             max_contributions_per_partition=1.5,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT]),
        dict(
            testcase_name='min_value is not set, max_value is set SUM',
            error_msg='min_value and max_value should be both set or both None',
            min_value=None,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            max_contributions=None,
            metrics=[pipeline_dp.Metrics.SUM]),
        dict(
            testcase_name='unspecified max_value SUM',
            error_msg='min_value and max_value should be both set or both None',
            min_value=1,
            max_value=None,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            max_contributions=None,
            metrics=[pipeline_dp.Metrics.SUM]),
        dict(
            testcase_name='unspecified min_value MEAN',
            error_msg='min_value and max_value should be both set or both None',
            min_value=None,
            max_value=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            max_contributions=None,
            metrics=[pipeline_dp.Metrics.MEAN]),
        dict(
            testcase_name='unspecified max_value MEAN',
            error_msg='min_value and max_value should be both set or both None',
            min_value=1,
            max_value=None,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            max_contributions=None,
            metrics=[pipeline_dp.Metrics.MEAN]),
        dict(testcase_name='min_value > max_value',
             error_msg='must be equal to or greater',
             min_value=2,
             max_value=1,
             max_partitions_contributed=1,
             max_contributions_per_partition=1,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.SUM]),
        dict(testcase_name='max_contrib and max_partitions are set',
             error_msg='only one in max_contributions or',
             min_value=0,
             max_value=1,
             max_partitions_contributed=1,
             max_contributions_per_partition=1,
             max_contributions=1,
             metrics=[pipeline_dp.Metrics.SUM]),
        dict(testcase_name='max_partitions_contributed not set',
             error_msg='either none or both',
             min_value=0,
             max_value=1,
             max_partitions_contributed=None,
             max_contributions_per_partition=1,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.SUM]),
        dict(testcase_name='contributions not set',
             error_msg='either max_contributions must',
             min_value=0,
             max_value=1,
             max_partitions_contributed=None,
             max_contributions_per_partition=None,
             max_contributions=None,
             metrics=[pipeline_dp.Metrics.SUM]),
        dict(testcase_name='vector sum computed with scalar sum',
             error_msg=
             'vector sum can not be computed together with scalar metrics',
             min_value=0,
             max_value=1,
             max_partitions_contributed=1,
             max_contributions_per_partition=1,
             max_contributions=1,
             metrics=[pipeline_dp.Metrics.VECTOR_SUM, pipeline_dp.Metrics.SUM]),
        dict(testcase_name='vector sum computed with scalar mean',
             error_msg=
             'vector sum can not be computed together with scalar metrics',
             min_value=0,
             max_value=1,
             max_partitions_contributed=1,
             max_contributions_per_partition=1,
             max_contributions=1,
             metrics=[pipeline_dp.Metrics.VECTOR_SUM,
                      pipeline_dp.Metrics.MEAN]),
        dict(testcase_name='vector sum computed with scalar variance',
             error_msg=
             'vector sum can not be computed together with scalar metrics',
             min_value=0,
             max_value=1,
             max_partitions_contributed=1,
             max_contributions_per_partition=1,
             max_contributions=1,
             metrics=[
                 pipeline_dp.Metrics.VECTOR_SUM, pipeline_dp.Metrics.VARIANCE
             ]),
    )
    def test_check_invalid_bounding_params(self, error_msg, min_value,
                                           max_value,
                                           max_partitions_contributed,
                                           max_contributions_per_partition,
                                           max_contributions, metrics):
        with self.assertRaisesRegex(ValueError, error_msg):
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=1e-10)
            engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                          backend=pipeline_dp.LocalBackend())
            engine.aggregate(
                [0],
                pipeline_dp.AggregateParams(
                    noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                    max_partitions_contributed=max_partitions_contributed,
                    max_contributions_per_partition=
                    max_contributions_per_partition,
                    min_value=min_value,
                    max_value=max_value,
                    max_contributions=max_contributions,
                    metrics=metrics), self._get_default_extractors())

    @parameterized.named_parameters(
        dict(testcase_name='used deprecated parameter low',
             error_msg='use min_value instead of low',
             deprecated_param={'low': 0}),
        dict(testcase_name='used deprecated parameter high',
             error_msg='use max_value instead of high',
             deprecated_param={'high': 1}),
        dict(testcase_name='used deprecated parameter public_partitions',
             error_msg='public_partitions is deprecated',
             deprecated_param={'public_partitions': [0]}),
    )
    def test_check_deprecated_params(self, error_msg, deprecated_param):
        with self.assertRaisesRegex(ValueError, error_msg):
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=1e-10)
            engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                          backend=pipeline_dp.LocalBackend())
            engine.aggregate([0],
                             pipeline_dp.AggregateParams(
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                 max_partitions_contributed=1,
                                 max_contributions_per_partition=1,
                                 **deprecated_param,
                                 max_contributions=1,
                                 metrics=[pipeline_dp.Metrics.COUNT]),
                             self._get_default_extractors())

    @parameterized.named_parameters(
        dict(
            testcase_name=
            'min_sum_per_partition is set, max_sum_per_partition is not set',
            error_msg=
            'min_sum_per_partition and max_sum_per_partition should be both set or both None',
            min_sum_per_partition=0,
            max_sum_per_partition=None,
            metrics=[pipeline_dp.Metrics.SUM]),
        dict(
            testcase_name=
            'min_sum_per_partition is not set, max_sum_per_partition is set',
            error_msg=
            'min_sum_per_partition and max_sum_per_partition should be both set or both None',
            min_sum_per_partition=None,
            max_sum_per_partition=1,
            metrics=[pipeline_dp.Metrics.SUM]),
        dict(
            testcase_name='min_sum_per_partition > max_sum_per_partition',
            error_msg=
            'max_sum_per_partition must be equal to or greater than min_sum_per_partition',
            min_sum_per_partition=1,
            max_sum_per_partition=0,
            metrics=[pipeline_dp.Metrics.SUM]),
        dict(testcase_name='min_sum_per_partition not compatible with mean',
             error_msg='min_sum_per_partition is not compatible with metrics',
             min_sum_per_partition=0,
             max_sum_per_partition=1,
             metrics=[pipeline_dp.Metrics.MEAN]),
        dict(testcase_name='min_sum_per_partition not compatible with variance',
             error_msg='min_sum_per_partition is not compatible with metrics',
             min_sum_per_partition=0,
             max_sum_per_partition=1,
             metrics=[pipeline_dp.Metrics.VARIANCE]),
        dict(testcase_name='all bounds per partition not set, metrics is SUM',
             error_msg='bounds per partition are required',
             min_sum_per_partition=None,
             max_sum_per_partition=None,
             metrics=[pipeline_dp.Metrics.SUM]),
        dict(testcase_name='all bounds per partition not set, metrics is MEAN',
             error_msg='bounds per partition are required',
             min_sum_per_partition=None,
             max_sum_per_partition=None,
             metrics=[pipeline_dp.Metrics.MEAN]),
        dict(testcase_name=
             'all bounds per partition not set, metrics is VARIANCE',
             error_msg='bounds per partition are required',
             min_sum_per_partition=None,
             max_sum_per_partition=None,
             metrics=[pipeline_dp.Metrics.VARIANCE]),
    )
    def test_check_sum_per_partition_params(self, error_msg,
                                            min_sum_per_partition,
                                            max_sum_per_partition, metrics):
        with self.assertRaisesRegex(ValueError, error_msg):
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=1e-10)
            engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                          backend=pipeline_dp.LocalBackend())
            engine.aggregate([0],
                             pipeline_dp.AggregateParams(
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                 min_value=None,
                                 max_value=None,
                                 max_partitions_contributed=1,
                                 max_contributions_per_partition=1,
                                 min_sum_per_partition=min_sum_per_partition,
                                 max_sum_per_partition=max_sum_per_partition,
                                 max_contributions=1,
                                 metrics=metrics),
                             self._get_default_extractors())

    @parameterized.named_parameters(
        dict(testcase_name='both metrics and custom_combiners are set',
             error_msg='Custom combiners can not be used with standard metrics',
             metrics=[pipeline_dp.Metrics.SUM],
             custom_combiners=[0],
             contribution_bounds_already_enforced=None),
        dict(
            testcase_name=
            'metrics is PRIVACY_ID_COUNT and contribution_bounds_already_enforced is True',
            error_msg=
            'PRIVACY_ID_COUNT when contribution_bounds_already_enforced is set to True',
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            custom_combiners=None,
            contribution_bounds_already_enforced=True),
    )
    def test_check_metrics_with_parameters_not_allowed(
            self, error_msg, metrics, custom_combiners,
            contribution_bounds_already_enforced):
        with self.assertRaisesRegex(ValueError, error_msg):
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=1e-10)
            engine = pipeline_dp.DPEngine(budget_accountant=budget_accountant,
                                          backend=pipeline_dp.LocalBackend())
            engine.aggregate([0],
                             pipeline_dp.AggregateParams(
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                                 min_value=1,
                                 max_value=1,
                                 metrics=metrics,
                                 custom_combiners=custom_combiners,
                                 contribution_bounds_already_enforced=
                                 contribution_bounds_already_enforced),
                             self._get_default_extractors())
