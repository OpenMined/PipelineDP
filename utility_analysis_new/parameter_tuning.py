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

import pipeline_dp
import utility_analysis_new
from pipeline_dp import pipeline_backend
from pipeline_dp import input_validators
from utility_analysis_new import combiners
from utility_analysis_new import histograms
from utility_analysis_new import metrics
from utility_analysis_new import utility_analysis

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
from enum import Enum
import numpy as np


@dataclass
class UtilityAnalysisRun:
    params: utility_analysis.UtilityAnalysisOptions
    result: metrics.AggregateErrorMetrics


class MinimizingFunction(Enum):
    ABSOLUTE_ERROR = 'absolute_error'
    RELATIVE_ERROR = 'relative_error'


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
    """Options for tuning process.

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
    """
    epsilon: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams
    function_to_minimize: Union[MinimizingFunction, Callable]
    parameters_to_tune: ParametersToTune

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
        utility_analysis_results: the results of all utility analysis runs that
          were performed during the tuning process.
    """
    options: TuneOptions
    contribution_histograms: histograms.ContributionHistograms
    utility_analysis_parameters: utility_analysis_new.dp_engine.MultiParameterConfiguration
    index_best: int
    utility_analysis_results: List[metrics.AggregateErrorMetrics]


def _find_candidate_parameters(
    hist: histograms.ContributionHistograms,
    parameters_to_tune: ParametersToTune
) -> utility_analysis_new.dp_engine.MultiParameterConfiguration:
    """Uses some heuristics to find (hopefully) good enough parameters."""
    # TODO: decide where to put QUANTILES_TO_USE, maybe TuneOptions?
    QUANTILES_TO_USE = [0.9, 0.95, 0.98, 0.99, 0.995]
    l0_candidates = linf_candidates = None

    def _find_candidates(histogram: histograms.Histogram) -> List:
        candidates = histogram.quantiles(QUANTILES_TO_USE)
        candidates.append(histogram.max_value)
        candidates = list(set(candidates))  # remove duplicates
        candidates.sort()
        return candidates

    if parameters_to_tune.max_partitions_contributed:
        l0_candidates = _find_candidates(hist.cross_partition_histogram)

    if parameters_to_tune.max_contributions_per_partition:
        linf_candidates = _find_candidates(hist.per_partition_histogram)

    l0_bounds = linf_bounds = None

    if l0_candidates and linf_candidates:
        l0_bounds, linf_bounds = [], []
        for l0 in l0_candidates:
            for linf in linf_candidates:
                l0_bounds.append(l0)
                linf_bounds.append(linf)
    elif l0_candidates:
        l0_bounds = l0_candidates
    elif linf_candidates:
        linf_bounds = linf_candidates
    else:
        assert False, "Nothing to tune."

    return utility_analysis_new.dp_engine.MultiParameterConfiguration(
        max_partitions_contributed=l0_bounds,
        max_contributions_per_partition=linf_bounds)


def _convert_utility_analysis_to_tune_result(
        utility_analysis_result: Tuple, tune_options: TuneOptions,
        run_configurations: utility_analysis_new.dp_engine.
    MultiParameterConfiguration, use_public_partitions: bool,
        contribution_histograms: histograms.ContributionHistograms
) -> TuneResult:
    assert len(utility_analysis_result) == run_configurations.size
    # TODO(dvadym): implement relative error.
    # TODO(dvadym): take into consideration partition selection from private
    # partition selection.
    assert tune_options.function_to_minimize == MinimizingFunction.ABSOLUTE_ERROR

    index_best = np.argmin([
        ae.aggregate_error_metrics.absolute_rmse()
        for ae in utility_analysis_result
    ])

    return TuneResult(tune_options, contribution_histograms, run_configurations,
                      index_best, utility_analysis_result)


def tune(col,
         backend: pipeline_backend.PipelineBackend,
         contribution_histograms: histograms.ContributionHistograms,
         options: TuneOptions,
         data_extractors: pipeline_dp.DataExtractors,
         public_partitions=None,
         return_utility_analysis_per_partition: bool = False) -> TuneResult:
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
          from elements of 'col'.
        public_partitions: A collection of partition keys that will be present
          in the result. If not provided, tuning will be performed assuming
          private partition selection is used.
        return_utility_analysis_per_partition: todo
    Returns:
        1 element collection which contains TuneResult.
    """
    _check_tune_args(options)

    candidates = _find_candidate_parameters(contribution_histograms,
                                            options.parameters_to_tune)

    utility_analysis_options = utility_analysis.UtilityAnalysisOptions(
        options.epsilon,
        options.delta,
        options.aggregate_params,
        multi_param_configuration=candidates)
    result = utility_analysis.perform_utility_analysis(
        col, backend, utility_analysis_options, data_extractors,
        public_partitions, return_utility_analysis_per_partition)
    if return_utility_analysis_per_partition:
        utility_analysis_result, utility_analysis_result_per_partition = result
    else:
        utility_analysis_result = result
    use_public_partitions = public_partitions is not None
    utility_analysis_result = backend.map(
        utility_analysis_result,
        lambda result: _convert_utility_analysis_to_tune_result(
            result, options, candidates, use_public_partitions,
            contribution_histograms), "To Tune result")
    if return_utility_analysis_per_partition:
        return utility_analysis_result, utility_analysis_result_per_partition
    return utility_analysis_result


def _check_tune_args(options: TuneOptions):
    if options.aggregate_params.metrics != [pipeline_dp.Metrics.COUNT]:
        raise NotImplementedError("Tuning is supported only for Count.")

    if options.function_to_minimize != MinimizingFunction.ABSOLUTE_ERROR:
        raise NotImplementedError(
            f"Only {MinimizingFunction.ABSOLUTE_ERROR} is implemented.")
