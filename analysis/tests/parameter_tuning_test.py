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
from unittest.mock import patch
from typing import List

import pipeline_dp
from analysis import metrics
from analysis import parameter_tuning
from analysis.parameter_tuning import ParametersSearchStrategy
from pipeline_dp.dataset_histograms import histograms
from pipeline_dp.dataset_histograms import computing_histograms


def _get_aggregate_params():
    # Limit contributions to 1 per partition, contribution error will be half of the count.
    return pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1)


def _get_tune_options():
    return parameter_tuning.TuneOptions(
        epsilon=1,
        delta=1e-10,
        aggregate_params=_get_aggregate_params(),
        function_to_minimize=parameter_tuning.MinimizingFunction.ABSOLUTE_ERROR,
        parameters_to_tune=parameter_tuning.ParametersToTune(True, True),
        number_of_parameter_candidates=3)


class ParameterTuning(parameterized.TestCase):

    @parameterized.parameters(
        (True, True, pipeline_dp.Metrics.COUNT, [1, 1, 2, 2, 6, 6
                                                ], [3, 6, 3, 6, 3, 6]),
        (False, True, pipeline_dp.Metrics.COUNT, None, [3, 6]),
        (True, False, pipeline_dp.Metrics.COUNT, [1, 2, 6], None),
        (True, True, pipeline_dp.Metrics.PRIVACY_ID_COUNT, [1, 2, 6], None),
    )
    def test_find_candidate_parameters_quantiles_strategy(
        self,
        tune_max_partitions_contributed: bool,
        tune_max_contributions_per_partition: bool,
        metric: pipeline_dp.Metrics,
        expected_max_partitions_contributed: List,
        expected_max_contributions_per_partition: List,
    ):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.quantiles = mock.Mock(return_value=[1, 1, 2])
        mock_l0_histogram.max_value = mock.Mock(return_value=6)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.quantiles = mock.Mock(return_value=[3, 6, 6])
        mock_linf_histogram.max_value = mock.Mock(return_value=6)

        mock_histograms = histograms.DatasetHistograms(mock_l0_histogram, None,
                                                       mock_linf_histogram,
                                                       None, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=tune_max_partitions_contributed,
            max_contributions_per_partition=tune_max_contributions_per_partition
        )

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            metric,
            ParametersSearchStrategy.QUANTILES,
            max_candidates=100)
        self.assertEqual(expected_max_partitions_contributed,
                         candidates.max_partitions_contributed)
        self.assertEqual(expected_max_contributions_per_partition,
                         candidates.max_contributions_per_partition)

    def test_find_candidate_parameters_maximum_number_of_candidates_is_respected_when_both_parameters_needs_to_be_tuned(
            self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.quantiles = mock.Mock(return_value=[1, 2, 3])
        mock_l0_histogram.max_value = mock.Mock(return_value=6)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.quantiles = mock.Mock(return_value=[4, 5, 6])
        mock_linf_histogram.max_value = mock.Mock(return_value=6)

        mock_histograms = histograms.DatasetHistograms(mock_l0_histogram, None,
                                                       mock_linf_histogram,
                                                       None, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=True,
            max_contributions_per_partition=True)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            pipeline_dp.Metrics.COUNT,
            ParametersSearchStrategy.QUANTILES,
            max_candidates=5)
        self.assertEqual([1, 1, 2, 2], candidates.max_partitions_contributed)
        self.assertEqual([4, 5, 4, 5],
                         candidates.max_contributions_per_partition)

    def test_find_candidate_parameters_more_candidates_for_l_0_when_not_so_many_l_inf_candidates(
            self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.quantiles = mock.Mock(return_value=[1, 2, 3, 4, 5])
        mock_l0_histogram.max_value = mock.Mock(return_value=6)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.quantiles = mock.Mock(return_value=[6, 7])
        mock_linf_histogram.max_value = mock.Mock(return_value=6)

        mock_histograms = histograms.DatasetHistograms(mock_l0_histogram, None,
                                                       mock_linf_histogram,
                                                       None, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=True,
            max_contributions_per_partition=True)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            pipeline_dp.Metrics.COUNT,
            ParametersSearchStrategy.QUANTILES,
            max_candidates=9)
        # sqrt(9) = 3, but l_inf has only 2 quantiles, therefore for l_0 we can
        # take 9 / 2 = 4 quantiles, we take first 4 quantiles (1, 2, 3, 4).
        # Addition of max_value (6) to l_inf does not change anything because
        # l_inf set already contains 6.
        self.assertEqual([1, 1, 2, 2, 3, 3, 4, 4],
                         candidates.max_partitions_contributed)
        self.assertEqual([6, 7, 6, 7, 6, 7, 6, 7],
                         candidates.max_contributions_per_partition)

    def test_find_candidate_parameters_more_candidates_for_l_inf_when_not_so_many_l_0_candidates(
            self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.quantiles = mock.Mock(return_value=[1])
        mock_l0_histogram.max_value = mock.Mock(return_value=8)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.quantiles = mock.Mock(return_value=[3, 4, 5, 6, 7])
        mock_linf_histogram.max_value = mock.Mock(return_value=8)

        mock_histograms = histograms.DatasetHistograms(mock_l0_histogram, None,
                                                       mock_linf_histogram,
                                                       None, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=True,
            max_contributions_per_partition=True)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            pipeline_dp.Metrics.COUNT,
            ParametersSearchStrategy.QUANTILES,
            max_candidates=10)
        # sqrt(10) = 3, but l_0 has only 2 quantiles (1 and 8 -- max_value),
        # therefore for l_inf we can take 10 / 2 = 5 quantiles.
        self.assertEqual([1, 1, 1, 1, 1, 8, 8, 8, 8, 8],
                         candidates.max_partitions_contributed)
        self.assertEqual([3, 4, 5, 6, 7, 3, 4, 5, 6, 7],
                         candidates.max_contributions_per_partition)

    @parameterized.named_parameters(
        dict(testcase_name='max_value=1, returns [1]',
             max_value=1,
             max_candidates=1000,
             expected_candidates=[1]),
        dict(testcase_name='max_candidates=1, returns [1]',
             max_value=1000,
             max_candidates=1,
             expected_candidates=[1]),
        dict(testcase_name='max_candidates=2, returns 1 and max_value',
             max_value=1003,
             max_candidates=2,
             expected_candidates=[1, 1003]),
        dict(testcase_name='max_candidates is equal to max_value, returns '
             'all possible candidates',
             max_value=10,
             max_candidates=10,
             expected_candidates=list(range(1, 11))),
        dict(
            testcase_name='max_candidates is larger than max_value, returns all'
            ' possible candidates up to max_value',
            max_value=10,
            max_candidates=100,
            expected_candidates=list(range(1, 11))),
        dict(
            testcase_name='max_candidates is smaller than max_value, returns '
            'logarithmic subset of values and last value is '
            'max_value',
            max_value=1000,
            max_candidates=5,
            # ceil(1000^(i / 4)), where i in [0, 1, 2, 3, 4]
            expected_candidates=[1, 6, 32, 178, 1000]))
    def test_find_candidate_parameters_constant_relative_step_strategy(
            self, max_value, max_candidates, expected_candidates):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.max_value = mock.Mock(return_value=max_value)

        mock_histograms = histograms.DatasetHistograms(mock_l0_histogram, None,
                                                       None, None, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=True,
            max_contributions_per_partition=False)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            pipeline_dp.Metrics.COUNT,
            ParametersSearchStrategy.CONSTANT_RELATIVE_STEP,
            max_candidates=max_candidates)

        self.assertEqual(expected_candidates,
                         candidates.max_partitions_contributed)

    @parameterized.named_parameters(
        dict(
            testcase_name='COUNT',
            metric=pipeline_dp.Metrics.COUNT,
            expected_generate_linf=True,
        ),
        dict(
            testcase_name='PRIVACY_ID_COUNT',
            metric=pipeline_dp.Metrics.PRIVACY_ID_COUNT,
            expected_generate_linf=False,
        ),
        dict(
            testcase_name='No metric (select partition)',
            metric=None,
            expected_generate_linf=False,
        ))
    @patch('analysis.parameter_tuning._find_candidates_constant_relative_step')
    def test_find_candidate_parameters_generate_linf(
            self, mock_find_candidate_from_histogram, metric,
            expected_generate_linf):
        mock_l0_histogram = histograms.Histogram(
            histograms.HistogramType.L0_CONTRIBUTIONS, None)
        mock_linf_histogram = histograms.Histogram(
            histograms.HistogramType.LINF_CONTRIBUTIONS, None)
        mock_histograms = histograms.DatasetHistograms(mock_l0_histogram, None,
                                                       mock_linf_histogram,
                                                       None, None, None)

        mock_find_candidate_from_histogram.return_value = [1, 2]

        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=True,
            max_contributions_per_partition=True)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            metric,
            ParametersSearchStrategy.CONSTANT_RELATIVE_STEP,
            max_candidates=100)

        mock_find_candidate_from_histogram.assert_any_call(
            mock_l0_histogram, mock.ANY)
        if expected_generate_linf:
            self.assertEqual(candidates.max_partitions_contributed,
                             [1, 1, 2, 2])
            self.assertEqual(candidates.max_contributions_per_partition,
                             [1, 2, 1, 2])
            mock_find_candidate_from_histogram.assert_any_call(
                mock_linf_histogram, mock.ANY)
        else:
            self.assertEqual(candidates.max_partitions_contributed, [1, 2])
            self.assertIsNone(candidates.max_contributions_per_partition)

    def test_tune_count(self):
        # Arrange.
        # Generate dataset, with 10 privacy units, each of them contribute to
        # the same partition.
        input = [(i % 10, f"pk0") for i in range(10)]
        public_partitions = [f"pk{i}" for i in range(10)]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: 0)

        contribution_histograms = list(
            computing_histograms.compute_dataset_histograms(
                input, data_extractors, pipeline_dp.LocalBackend()))[0]

        tune_options = _get_tune_options()

        # Act.
        result = parameter_tuning.tune(input, pipeline_dp.LocalBackend(),
                                       contribution_histograms, tune_options,
                                       data_extractors, public_partitions)

        # Assert.
        tune_result, per_partition_utility_analysis = result
        per_partition_utility_analysis = list(per_partition_utility_analysis)
        self.assertLen(per_partition_utility_analysis, 10)

        tune_result = list(tune_result)[0]

        self.assertEqual(tune_options, tune_result.options)
        self.assertEqual(contribution_histograms,
                         tune_result.contribution_histograms)
        utility_reports = tune_result.utility_reports
        self.assertLen(utility_reports, 1)
        self.assertIsInstance(utility_reports[0], metrics.UtilityReport)
        self.assertLen(utility_reports[0].metric_errors, 1)
        self.assertEqual(utility_reports[0].metric_errors[0].metric,
                         pipeline_dp.Metrics.COUNT)

    def test_select_partitions(self):
        # Arrange.
        # Generate dataset, with 10 privacy units, 5 of them contribute to
        # pk0 and pk1 and 5 to pk0 only.
        input = [(i % 10, f"pk{i//10}") for i in range(15)]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: 0)

        contribution_histograms = list(
            computing_histograms.compute_dataset_histograms(
                input, data_extractors, pipeline_dp.LocalBackend()))[0]

        tune_options = _get_tune_options()
        # Setting metrics to empty list makes running only partition selectoin.
        tune_options.aggregate_params.metrics = []

        # Act.
        result = parameter_tuning.tune(input, pipeline_dp.LocalBackend(),
                                       contribution_histograms, tune_options,
                                       data_extractors)

        # Assert.
        tune_result, per_partition_utility_analysis = result
        per_partition_utility_analysis = list(per_partition_utility_analysis)
        self.assertLen(per_partition_utility_analysis, 4)

        tune_result = list(tune_result)[0]

        self.assertEqual(tune_options, tune_result.options)
        self.assertEqual(contribution_histograms,
                         tune_result.contribution_histograms)
        utility_reports = tune_result.utility_reports
        self.assertLen(utility_reports, 2)
        self.assertIsInstance(utility_reports[0], metrics.UtilityReport)
        self.assertIsNone(utility_reports[0].metric_errors)
        self.assertEqual(
            utility_reports[0].partitions_info.num_dataset_partitions, 2)
        self.assertAlmostEqual(
            utility_reports[0].partitions_info.kept_partitions.mean, 1.93e-7)

    def test_tune_privacy_id_count(self):
        # Arrange.
        input = [(i % 10, f"pk{i/10}", i) for i in range(10)]
        public_partitions = [f"pk{i}" for i in range(10)]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: x[2])

        contribution_histograms = list(
            computing_histograms.compute_dataset_histograms(
                input, data_extractors, pipeline_dp.LocalBackend()))[0]

        tune_options = _get_tune_options()
        tune_options.aggregate_params.metrics = [
            pipeline_dp.Metrics.PRIVACY_ID_COUNT
        ]

        # Act.
        result, _ = parameter_tuning.tune(input, pipeline_dp.LocalBackend(),
                                          contribution_histograms, tune_options,
                                          data_extractors, public_partitions)

        # Assert.
        result = list(result)[0]

        self.assertEqual(tune_options, result.options)
        self.assertEqual(contribution_histograms,
                         result.contribution_histograms)
        utility_reports = result.utility_reports
        self.assertIsInstance(utility_reports[0], metrics.UtilityReport)
        self.assertLen(utility_reports[0].metric_errors, 1)
        self.assertEqual(utility_reports[0].metric_errors[0].metric,
                         pipeline_dp.Metrics.PRIVACY_ID_COUNT)

    @parameterized.named_parameters(
        dict(testcase_name="Select partition and public partition",
             error_msg="Empty metrics means tuning of partition selection but"
             " public partitions were provided",
             metrics=[],
             is_public_partitions=True),
        dict(testcase_name="Multiple metrics",
             error_msg="Tuning supports only one metric",
             metrics=[
                 pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT
             ],
             is_public_partitions=True),
        dict(
            testcase_name="Mean is not supported",
            error_msg="Tuning is supported only for Count and Privacy id count",
            metrics=[pipeline_dp.Metrics.MEAN],
            is_public_partitions=False),
    )
    def test_tune_params_validation(self, error_msg,
                                    metrics: List[pipeline_dp.Metric],
                                    is_public_partitions: bool):
        tune_options = _get_tune_options()
        tune_options.aggregate_params.metrics = metrics
        contribution_histograms = histograms.DatasetHistograms(
            None, None, None, None, None, None)
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda _: 0, partition_extractor=lambda _: 0)
        public_partitions = [1] if is_public_partitions else None
        with self.assertRaisesRegex(ValueError, error_msg):
            parameter_tuning.tune(input, pipeline_dp.LocalBackend(),
                                  contribution_histograms, tune_options,
                                  data_extractors, public_partitions)


if __name__ == '__main__':
    absltest.main()
