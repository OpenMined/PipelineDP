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
import math

import pipeline_dp
from pipeline_dp import pipeline_backend
from pipeline_dp import input_validators
from pipeline_dp.dataset_histograms import histograms
import analysis
from analysis import metrics
from analysis import utility_analysis

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
from enum import Enum
import numpy as np

from pipeline_dp import private_contribution_bounds


class MinimizingFunction(Enum):
    ABSOLUTE_ERROR = 'absolute_error'
    RELATIVE_ERROR = 'relative_error'


class ParametersSearchStrategy(Enum):
    """Strategy types for selecting candidate parameters."""

    # Picks up candidates that correspond tp a predefined list of quantiles.
    QUANTILES = 1
    # Candidates are a sequence starting from 1 where relative difference
    # between two neighbouring elements is (almost) the same.
    CONSTANT_RELATIVE_STEP = 2


@dataclass
class ParametersToTune:
    """Contains parameters to tune."""
    max_partitions_contributed: bool = False
    max_contributions_per_partition: bool = False
    min_sum_per_partition: bool = False
    max_sum_per_partition: bool = False

    def __post_init__(self):
        if not any(dataclasses.asdict(self).values()):
            raise ValueError("ParametersToTune must have at least 1 parameter "
                             "to tune.")


@dataclass
class TuneOptions:
    """Options for the tuning process.

    Note that parameters that are not tuned (e.g. metrics, noise kind) are taken
    from aggregate_params.

    Attributes:
        epsilon, delta: differential privacy budget for aggregations for which
          tuning is performed.
        aggregate_params: parameters of aggregation.
        function_to_minimize: which function of the error to minimize. In case
          if this argument is a callable, it should take 1 argument of type
          AggregateErrorMetrics and return float.
        parameters_to_tune: specifies which parameters to tune.
        partitions_sampling_prob: the probability with which each partition
          will be sampled before running tuning. It is useful for speed-up
          computations on the large datasets.
        pre_aggregated_data: when True the input data is already pre-aggregated,
          otherwise the input data are raw. Preaggregated data also can be
          sampled.
        parameters_search_strategy: specifies how to select candidates for
          parameters.
        number_of_parameter_candidates: how many candidates to generate for
          parameter tuning. This is an upper bound, there can be fewer
          candidates generated.

    """
    epsilon: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams
    function_to_minimize: Union[MinimizingFunction, Callable]
    parameters_to_tune: ParametersToTune
    partitions_sampling_prob: float = 1
    pre_aggregated_data: bool = False
    parameters_search_strategy: ParametersSearchStrategy = ParametersSearchStrategy.QUANTILES
    number_of_parameter_candidates: int = 100

    def __post_init__(self):
        input_validators.validate_epsilon_delta(self.epsilon, self.delta,
                                                "TuneOptions")


@dataclass
class TuneResult:
    """Represents tune results.

    Attributes:
        options: input options for tuning.
        contribution_histograms: histograms of privacy id contributions.
        utility_analysis_parameters: contains tune parameter values for which
        utility analysis ran.
        index_best: index of the recommended according to minimizing function
         (best) configuration in utility_analysis_parameters. Note, that those
          parameters might not necessarily be optimal, since finding the optimal
          parameters is not always feasible.
        utility_reports: the results of all utility analysis runs that
          were performed during the tuning process.
    """
    options: TuneOptions
    contribution_histograms: histograms.DatasetHistograms
    utility_analysis_parameters: analysis.MultiParameterConfiguration
    index_best: int
    utility_reports: List[metrics.UtilityReport]


def _find_candidate_parameters(
        hist: histograms.DatasetHistograms,
        parameters_to_tune: ParametersToTune, metric: pipeline_dp.Metrics,
        strategy, max_candidates) -> analysis.MultiParameterConfiguration:
    """Finds candidates for l0 and/or l_inf parameters.

    Args:
        strategy: determines the strategy how to select candidates, see comments
          to enum values for full description of the respective strategies.
        max_candidates: how many candidates ((l0, linf) pairs) can be in the
        output. Note that output can contain fewer candidates. 100 is default
        heuristically chosen value, better to adjust it for your use-case.
    """
    if strategy == ParametersSearchStrategy.QUANTILES:
        find_candidates_func = _find_candidates_quantiles
    elif strategy == ParametersSearchStrategy.CONSTANT_RELATIVE_STEP:
        find_candidates_func = _find_candidates_constant_relative_step
    else:
        raise ValueError("Unknown strategy for candidate parameters search.")

    calculate_l0_param = parameters_to_tune.max_partitions_contributed
    calculate_linf_param = (parameters_to_tune.max_contributions_per_partition
                            and metric == pipeline_dp.Metrics.COUNT)
    l0_bounds = linf_bounds = None

    if calculate_l0_param and calculate_linf_param:
        max_candidates_per_parameter = int(math.sqrt(max_candidates))
        l0_candidates = find_candidates_func(hist.l0_contributions_histogram,
                                             max_candidates_per_parameter)
        linf_candidates = find_candidates_func(
            hist.linf_contributions_histogram, max_candidates_per_parameter)
        l0_bounds, linf_bounds = [], []
        for l0 in l0_candidates:
            for linf in linf_candidates:
                l0_bounds.append(l0)
                linf_bounds.append(linf)
    elif calculate_l0_param:
        l0_bounds = find_candidates_func(hist.l0_contributions_histogram,
                                         max_candidates)
    elif calculate_linf_param:
        linf_bounds = find_candidates_func(hist.linf_contributions_histogram,
                                           max_candidates)
    else:
        assert False, "Nothing to tune."

    return analysis.MultiParameterConfiguration(
        max_partitions_contributed=l0_bounds,
        max_contributions_per_partition=linf_bounds)


def _find_candidates_quantiles(histogram: histograms.Histogram,
                               max_candidates: int) -> List[int]:
    """Implementation of QUANTILES strategy."""
    quantiles_to_use = [0.9, 0.95, 0.98, 0.99, 0.995]
    candidates = histogram.quantiles(quantiles_to_use)
    candidates.append(histogram.max_value)
    candidates = list(set(candidates))  # remove duplicates
    candidates.sort()
    return candidates[:max_candidates]


def _find_candidates_constant_relative_step(histogram: histograms.Histogram,
                                            max_candidates: int) -> List[int]:
    """Implementation of CONSTANT_RELATIVE_STEP strategy."""
    max_value = histogram.max_value
    # relative step varies from 1% to 0.1%
    # because generate_possible_contribution_bounds generate bounds by changing
    # only up to first 3 digits, for example 100000, 101000, 102000... Then
    # relative step between neighbouring elements
    # varies (101000 - 100000) / 100000 = 0.01 and
    # (1000000 - 999000) / 999000 ~= 0.001.
    candidates = private_contribution_bounds.generate_possible_contribution_bounds(
        max_value)
    n_max_without_max_value = max_candidates - 1
    if len(candidates) > n_max_without_max_value:
        delta = len(candidates) / n_max_without_max_value
        candidates = [
            candidates[int(i * delta)] for i in range(n_max_without_max_value)
        ]
    if candidates[-1] != max_value:
        candidates.append(max_value)
    return candidates


def tune(col,
         backend: pipeline_backend.PipelineBackend,
         contribution_histograms: histograms.DatasetHistograms,
         options: TuneOptions,
         data_extractors: Union[pipeline_dp.DataExtractors,
                                pipeline_dp.PreAggregateExtractors],
         public_partitions=None,
         return_utility_analysis_per_partition: bool = False):
    """Tunes parameters.

    It works in the following way:
        1. Based on quantiles of privacy id contributions, candidates for
        contribution bounding parameters chosen.
        2. Utility analysis run for those parameters.
        3. The best parameter set is chosen according to
          options.minimizing_function.

    The result contains output metrics for all utility analysis which were
    performed.

    Args:
        col: collection where all elements are of the same type.
          contribution_histograms:
        backend: PipelineBackend with which the utility analysis will be run.
        contribution_histograms: contribution histograms that should be
         computed with compute_contribution_histograms().
        options: options for tuning.
        data_extractors: functions that extract needed pieces of information
          from elements of 'col'. In case if the analysis performed on
          pre-aggregated data, it should have type PreAggregateExtractors
          otherwise DataExtractors.
        public_partitions: A collection of partition keys that will be present
          in the result. If not provided, tuning will be performed assuming
          private partition selection is used.
        return_per_partition: if true, it returns tuple, with the 2nd element
          utility analysis per partitions.
    Returns:
        if return_per_partition == False:
            returns 1 element collection which contains TuneResult
        else returns tuple (1 element collection which contains TuneResult,
        a collection which contains utility analysis results per partition).
    """
    _check_tune_args(options)

    candidates = _find_candidate_parameters(
        contribution_histograms, options.parameters_to_tune,
        options.aggregate_params.metrics[0], options.parameters_search_strategy,
        options.number_of_parameter_candidates)

    utility_analysis_options = analysis.UtilityAnalysisOptions(
        epsilon=options.epsilon,
        delta=options.delta,
        aggregate_params=options.aggregate_params,
        multi_param_configuration=candidates,
        partitions_sampling_prob=options.partitions_sampling_prob,
        pre_aggregated_data=options.pre_aggregated_data)
    result = utility_analysis.perform_utility_analysis(
        col, backend, utility_analysis_options, data_extractors,
        public_partitions, return_utility_analysis_per_partition)
    if return_utility_analysis_per_partition:
        utility_result, per_partition_utility_result = result
    else:
        utility_result = result
    # utility_result: (UtilityReport)
    # per_partition_utility_result: (pk, (PerPartitionMetrics))
    use_public_partitions = public_partitions is not None

    utility_result = backend.to_list(utility_result, "To list")
    # 1 element collection with list[UtilityReport]
    utility_result = backend.map(
        utility_result,
        lambda result: _convert_utility_analysis_to_tune_result_new(
            result, options, candidates, use_public_partitions,
            contribution_histograms), "To Tune result")
    if return_utility_analysis_per_partition:
        return utility_result, per_partition_utility_result
    return utility_result


def _convert_utility_analysis_to_tune_result_new(
        utility_reports: Tuple[metrics.UtilityReport],
        tune_options: TuneOptions,
        run_configurations: analysis.MultiParameterConfiguration,
        use_public_partitions: bool,
        contribution_histograms: histograms.DatasetHistograms):
    assert len(utility_reports) == run_configurations.size
    # TODO(dvadym): implement relative error.
    # TODO(dvadym): take into consideration partition selection from private
    # partition selection.
    assert tune_options.function_to_minimize == MinimizingFunction.ABSOLUTE_ERROR

    # Sort utility reports by configuration index.
    utility_reports.sort(key=lambda e: e.configuration_index)

    index_best = -1  # not found
    # Find best index if there are metrics to compute. Absence of metrics to
    # compute means that this is SelectPartition analysis.
    if tune_options.aggregate_params.metrics:
        rmse = [
            ur.metric_errors[0].absolute_error.rmse for ur in utility_reports
        ]
        index_best = np.argmin(rmse)

    return TuneResult(tune_options,
                      contribution_histograms,
                      run_configurations,
                      index_best,
                      utility_reports=utility_reports)


def _check_tune_args(options: TuneOptions):
    # Check metrics to tune.
    metrics = options.aggregate_params.metrics
    if len(metrics) != 1:
        raise NotImplementedError(
            f"Tuning supports only one metrics, but {metrics} given.")
    if metrics[0] not in [
            pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT
    ]:
        raise NotImplementedError(
            f"Tuning is supported only for Count and Privacy id count, but {metrics[0]} given."
        )

    if options.function_to_minimize != MinimizingFunction.ABSOLUTE_ERROR:
        raise NotImplementedError(
            f"Only {MinimizingFunction.ABSOLUTE_ERROR} is implemented.")
