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
"""Parameter tuning test."""

from absl.testing import absltest
from absl.testing import parameterized
from unittest import mock
from typing import List

import pipeline_dp
from utility_analysis_new import histograms
from utility_analysis_new import metrics
from utility_analysis_new import parameter_tuning


def _get_aggregate_params():
    # Limit contributions to 1 per partition, contribution error will be half of the count.
    return pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1)


class ParameterTuning(parameterized.TestCase):

    @parameterized.parameters(
        (True, True, [1, 1, 2, 2, 6, 6], [3, 6, 3, 6, 3, 6]),
        (False, True, None, [3, 6]), (True, False, [1, 2, 6], None))
    def test_find_candidate_parameters(
        self,
        tune_max_partitions_contributed: bool,
        tune_max_contributions_per_partition: bool,
        expected_max_partitions_contributed: List,
        expected_max_contributions_per_partition: List,
    ):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.quantiles = mock.Mock(return_value=[1, 1, 2])
        setattr(mock_l0_histogram.__class__, 'max_value', 6)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.quantiles = mock.Mock(return_value=[3, 6, 6])

        mock_histograms = histograms.ContributionHistograms(
            mock_l0_histogram, mock_linf_histogram)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=tune_max_partitions_contributed,
            max_contributions_per_partition=tune_max_contributions_per_partition
        )

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms, parameters_to_tune)
        self.assertEqual(expected_max_partitions_contributed,
                         candidates.max_partitions_contributed)
        self.assertEqual(expected_max_contributions_per_partition,
                         candidates.max_contributions_per_partition)

    def test_tune(self):
        input = [(i % 10, f"pk{i/10}") for i in range(10)]
        public_partitions = [f"pk{i}" for i in range(10)]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: None)

        contribution_histograms = list(
            histograms.compute_contribution_histograms(
                input, data_extractors, pipeline_dp.LocalBackend()))[0]

        tune_options = parameter_tuning.TuneOptions(
            epsilon=1,
            delta=1e-10,
            aggregate_params=_get_aggregate_params(),
            function_to_minimize=parameter_tuning.MinimizingFunction.
            ABSOLUTE_ERROR,
            parameters_to_tune=parameter_tuning.ParametersToTune(True, True))
        tune_result = list(
            parameter_tuning.tune(input, pipeline_dp.LocalBackend(),
                                  contribution_histograms, tune_options,
                                  data_extractors, public_partitions))[0]

        self.assertEqual(tune_options, tune_result.options)
        self.assertEqual(contribution_histograms,
                         tune_result.contribution_histograms)
        self.assertIsInstance(tune_result.utility_analysis_results[0],
                              metrics.AggregateErrorMetrics)


if __name__ == '__main__':
    absltest.main()
