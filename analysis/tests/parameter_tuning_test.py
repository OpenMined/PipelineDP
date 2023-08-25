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
from pipeline_dp.dataset_histograms import histograms
from pipeline_dp.dataset_histograms import computing_histograms
from pipeline_dp.dataset_histograms.histograms import FrequencyBin


def _get_aggregate_params(metrics: List[pipeline_dp.Metric]):
    return pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=metrics,
        max_partitions_contributed=1,  # does not matter
        max_contributions_per_partition=1,  # does not matter
        min_value=0,  # does not matter
        max_value=1)  # does not matter


def _get_tune_options(
    metrics: List[pipeline_dp.Metric],
    parameters_to_tune: parameter_tuning.ParametersToTune = parameter_tuning.
    ParametersToTune(max_partitions_contributed=True,
                     max_contributions_per_partition=True)
) -> parameter_tuning.TuneOptions:
    return parameter_tuning.TuneOptions(
        epsilon=1,
        delta=1e-10,
        aggregate_params=_get_aggregate_params(metrics),
        function_to_minimize=parameter_tuning.MinimizingFunction.ABSOLUTE_ERROR,
        parameters_to_tune=parameters_to_tune,
        number_of_parameter_candidates=3)


def _frequency_bin(max_value: float = 0.0, lower: float = 0.0) -> FrequencyBin:
    return FrequencyBin(max=max_value, lower=lower, count=None, sum=None)


class ParameterTuning(parameterized.TestCase):

    def test_find_candidate_parameters_maximum_number_of_candidates_is_respected_when_both_parameters_needs_to_be_tuned(
            self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.max_value = mock.Mock(return_value=6)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.max_value = mock.Mock(return_value=3)

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
            max_candidates=5)
        self.assertEqual([1, 1, 6, 6], candidates.max_partitions_contributed)
        self.assertEqual([1, 3, 1, 3],
                         candidates.max_contributions_per_partition)

    def test_find_candidate_parameters_more_candidates_for_l_0_when_not_so_many_l_inf_candidates(
            self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.max_value = mock.Mock(return_value=4)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.max_value = mock.Mock(return_value=2)

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
            max_candidates=9)
        # sqrt(9) = 3, but l_inf has only 2 possible values,
        # therefore for l_0 we can take 9 / 2 = 4 values,
        # we take all 4 possible values (1, 2, 3, 4).
        self.assertEqual([1, 1, 2, 2, 3, 3, 4, 4],
                         candidates.max_partitions_contributed)
        self.assertEqual([1, 2, 1, 2, 1, 2, 1, 2],
                         candidates.max_contributions_per_partition)

    def test_find_candidate_parameters_more_candidates_for_l_inf_when_not_so_many_l_0_candidates(
            self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.max_value = mock.Mock(return_value=2)
        mock_linf_histogram = histograms.Histogram(None, None)
        mock_linf_histogram.max_value = mock.Mock(return_value=4)

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
            max_candidates=9)
        # sqrt(9) = 3, but l_0 has only 2 possible values,
        # therefore for l_inf we can take 9 / 2 = 4 values,
        # we take all 4 possible values (1, 2, 3, 4).
        self.assertEqual([1, 1, 1, 1, 2, 2, 2, 2],
                         candidates.max_partitions_contributed)
        self.assertEqual([1, 2, 3, 4, 1, 2, 3, 4],
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
    def test_find_candidate_parameters_count(self, max_value, max_candidates,
                                             expected_candidates):
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
            max_candidates=max_candidates)

        self.assertEqual(expected_candidates,
                         candidates.max_partitions_contributed)

    @parameterized.named_parameters(
        dict(testcase_name='bin_max_values=[1], returns [1]',
             bins=[_frequency_bin(max_value=1)],
             max_candidates=1000,
             expected_candidates=[1]),
        dict(testcase_name='max_candidates=1, returns max value of the first'
             ' bin',
             bins=[
                 _frequency_bin(max_value=0.1),
                 _frequency_bin(max_value=0.2),
                 _frequency_bin(max_value=0.3)
             ],
             max_candidates=1,
             expected_candidates=[0.1]),
        dict(testcase_name='max_candidates=2, returns max values of the first'
             ' and last bin',
             bins=[
                 _frequency_bin(max_value=0.1),
                 _frequency_bin(max_value=0.2),
                 _frequency_bin(max_value=0.3)
             ],
             max_candidates=2,
             expected_candidates=[0.1, 0.3]),
        dict(testcase_name='max_candidates is equal to number of bins, returns'
             ' all bin max values as candidates',
             bins=[
                 _frequency_bin(max_value=0.1),
                 _frequency_bin(max_value=0.2),
                 _frequency_bin(max_value=0.3)
             ],
             max_candidates=3,
             expected_candidates=[0.1, 0.2, 0.3]),
        dict(testcase_name='max_candidates is larger than number of bins,'
             ' returns all bin max values as candidates',
             bins=[
                 _frequency_bin(max_value=0.1),
                 _frequency_bin(max_value=0.2),
                 _frequency_bin(max_value=0.3)
             ],
             max_candidates=100,
             expected_candidates=[0.1, 0.2, 0.3]),
        dict(
            testcase_name='max_candidates is smaller than number of bins,'
            ' returns uniformly distributed subsample of bin'
            ' max values',
            bins=[_frequency_bin(max_value=i) for i in range(10)],
            max_candidates=5,
            # Takes each bin with step ((10 - 1) / (5 - 1) = 1.8), i.e.
            # [0, 2.25, 4.5, 6.75, 9], then rounds it, i.e. we get
            # [0, 2, 4, 7, 9] indices of bins to take, they equal to max
            # values of these bins
            expected_candidates=[0, 2, 4, 7, 9]),
    )
    def test_find_candidate_parameters_sum(self, bins, max_candidates,
                                           expected_candidates):
        mock_linf_sum_contributions_histogram = histograms.Histogram(None, bins)
        mock_histograms = histograms.DatasetHistograms(
            None, None, None, mock_linf_sum_contributions_histogram, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=False,
            min_sum_per_partition=False,
            max_sum_per_partition=True)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            pipeline_dp.Metrics.SUM,
            max_candidates=max_candidates)

        self.assertEqual(expected_candidates, candidates.max_sum_per_partition)
        self.assertEqual([0] * len(expected_candidates),
                         candidates.min_sum_per_partition)

    def test_find_candidate_parameters_min_sum_per_partition_is_not_supported(
            self):
        parameters_to_tune = parameter_tuning.ParametersToTune(
            min_sum_per_partition=True, max_sum_per_partition=True)

        with self.assertRaisesRegex(
                AssertionError,
                "Tuning of min_sum_per_partition is not supported yet"):
            parameter_tuning._find_candidate_parameters(
                hist=None,
                parameters_to_tune=parameters_to_tune,
                metric=pipeline_dp.Metrics.SUM,
                max_candidates=0)

    def test_find_candidate_parameters_sums_has_to_be_non_negative(self):
        mock_linf_sum_contributions_histogram = histograms.Histogram(
            None, [_frequency_bin(lower=-1.0)])
        mock_histograms = histograms.DatasetHistograms(
            None, None, None, mock_linf_sum_contributions_histogram, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            min_sum_per_partition=False, max_sum_per_partition=True)

        with self.assertRaisesRegex(
                AssertionError,
                "max_sum_per_partition should not contain negative sums"):
            parameter_tuning._find_candidate_parameters(
                hist=mock_histograms,
                parameters_to_tune=parameters_to_tune,
                metric=pipeline_dp.Metrics.SUM,
                max_candidates=0)

    def test_find_candidate_parameters_both_l0_and_linf_sum_to_be_tuned(self):
        mock_l0_histogram = histograms.Histogram(None, None)
        mock_l0_histogram.max_value = mock.Mock(return_value=6)
        mock_linf_sum_contributions_histogram = histograms.Histogram(
            None, [
                _frequency_bin(max_value=1),
                _frequency_bin(max_value=2),
                _frequency_bin(max_value=3)
            ])

        mock_histograms = histograms.DatasetHistograms(
            mock_l0_histogram, None, None,
            mock_linf_sum_contributions_histogram, None, None)
        parameters_to_tune = parameter_tuning.ParametersToTune(
            max_partitions_contributed=True,
            min_sum_per_partition=False,
            max_sum_per_partition=True)

        candidates = parameter_tuning._find_candidate_parameters(
            mock_histograms,
            parameters_to_tune,
            pipeline_dp.Metrics.SUM,
            max_candidates=5)
        self.assertEqual([1, 1, 6, 6], candidates.max_partitions_contributed)
        self.assertEqual([1, 3, 1, 3], candidates.max_sum_per_partition)
        self.assertEqual([0, 0, 0, 0], candidates.min_sum_per_partition)

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
            testcase_name='SUM',
            metric=pipeline_dp.Metrics.SUM,
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
            mock_histograms, parameters_to_tune, metric, max_candidates=100)

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

        tune_options = _get_tune_options(
            [pipeline_dp.Metrics.COUNT],
            parameter_tuning.ParametersToTune(
                max_partitions_contributed=True,
                max_contributions_per_partition=True))

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

    def test_tune_sum(self):
        # Arrange.
        # Generate dataset, with 10 privacy units, each of them contribute to
        # the same partition with value equal to its id.
        input = [(i, f"pk0", i) for i in range(10)]
        public_partitions = [f"pk{i}" for i in range(10)]
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: x[0],
            partition_extractor=lambda x: x[1],
            value_extractor=lambda x: x[2])

        contribution_histograms = list(
            computing_histograms.compute_dataset_histograms(
                input, data_extractors, pipeline_dp.LocalBackend()))[0]

        tune_options = _get_tune_options([pipeline_dp.Metrics.SUM],
                                         parameter_tuning.ParametersToTune(
                                             max_partitions_contributed=True,
                                             max_sum_per_partition=True))

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
                         pipeline_dp.Metrics.SUM)

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

        tune_options = _get_tune_options(metrics=[])

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

        tune_options = _get_tune_options(
            [pipeline_dp.Metrics.PRIVACY_ID_COUNT],
            parameter_tuning.ParametersToTune(max_partitions_contributed=True))

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
        dict(testcase_name="Mean is not supported",
             error_msg=
             "Tuning is supported only for Count, Privacy id count and Sum",
             metrics=[pipeline_dp.Metrics.MEAN],
             is_public_partitions=False),
    )
    def test_tune_params_validation(self, error_msg,
                                    metrics: List[pipeline_dp.Metric],
                                    is_public_partitions: bool):
        tune_options = _get_tune_options(metrics)
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
