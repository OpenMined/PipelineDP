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
"""DPEngine Test"""
import unittest

import pipeline_dp
from pipeline_dp import budget_accounting
from utility_analysis_new import dpengine


class DpEngine(unittest.TestCase):

    def _get_default_extractors(self) -> pipeline_dp.DataExtractors:
        return pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x,
            partition_extractor=lambda x: x,
            value_extractor=lambda x: x,
        )

    def test_utility_analysis_params(self):
        default_extractors = self._get_default_extractors()
        default_params = pipeline_dp.AggregateParams(
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            metrics=[pipeline_dp.Metrics.COUNT])
        params_with_custom_combiners = default_params
        params_with_custom_combiners.custom_combiners = lambda x: sum(x)
        params_with_unsupported_metric = default_params
        params_with_unsupported_metric.metrics = [pipeline_dp.Metrics.MEAN]
        params_with_contribution_bounds_already_enforced = default_params
        params_with_contribution_bounds_already_enforced.contribution_bounds_already_enforced = True

        test_cases = [
            {
                "desc": "custom combiners",
                "col": [0, 1, 2],
                "params": params_with_custom_combiners,
                "data_extractor": default_extractors,
                "public_partitions": [1]
            },
            {
                "desc": "non-supported metric",
                "col": [0, 1, 2],
                "params": params_with_unsupported_metric,
                "data_extractor": default_extractors,
                "public_partitions": [1]
            },
            {
                "desc": "private partitions",
                "col": [0, 1, 2],
                "params": params_with_unsupported_metric,
                "data_extractor": default_extractors,
                "public_partitions": None
            },
            {
                "desc": "contribution bounds are already enforced",
                "col": [0, 1, 2],
                "params": params_with_contribution_bounds_already_enforced,
                "data_extractor": default_extractors,
                "public_partitions": [1]
            },
        ]

        for test_case in test_cases:

            with self.assertRaises(Exception, msg=test_case["desc"]):
                budget_accountant = budget_accounting.NaiveBudgetAccountant(
                    total_epsilon=1, total_delta=1e-10)
                engine = dpengine.DPEngine(budget_accountant=budget_accountant,
                                           backend=pipeline_dp.LocalBackend())
                engine.aggregate(
                    test_case["col"],
                    test_case["params"],
                    test_case["data_extractor"],
                    public_partitions=test_case["public_partitions"])


if __name__ == '__main__':
    unittest.main()
