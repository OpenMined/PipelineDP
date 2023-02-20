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
from pipeline_dp import pipeline_backend
from pipeline_dp import input_validators
import analysis
from analysis import histograms
from analysis import metrics
from analysis import utility_analysis

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union
from enum import Enum
import numpy as np


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
          UtilityAnalysisResult and return float.
        parameters_to_tune: specifies which parameters to tune.
        partitions_sampling_prob: the probability with which each partition
        will be sampled before running tuning. It is useful for speed-up
        computations on the large datasets.
        pre_aggregated_data: when True the input data is already pre-aggregated,
        otherwise the input data are raw. Preaggregated data also can be
        sampled.
    """
    epsilon: float
    delta: float
    aggregate_params: pipeline_dp.AggregateParams
    function_to_minimize: Union[MinimizingFunction, Callable]
    parameters_to_tune: ParametersToTune
    partitions_sampling_prob: float = 1
    pre_aggregated_data: bool = False

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
    contribution_histograms: histograms.DatasetHistograms
    utility_analysis_parameters: analysis.MultiParameterConfiguration
    index_best: int
    utility_results: List[metrics.UtilityResult]


def _find_candidate_parameters(
        hist: histograms.DatasetHistograms,
        parameters_to_tune: ParametersToTune,
        metric: pipeline_dp.Metrics) -> analysis.MultiParameterConfiguration:
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
        l0_candidates = _find_candidates(hist.l0_contributions_histogram)

    if parameters_to_tune.max_contributions_per_partition and metric == pipeline_dp.Metrics.COUNT:
        linf_candidates = _find_candidates(hist.linf_contributions_histogram)

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

    return analysis.MultiParameterConfiguration(
        max_partitions_contributed=l0_bounds,
        max_contributions_per_partition=linf_bounds)


def _convert_utility_analysis_to_tune_result(
        utility_analysis_result: Tuple, tune_options: TuneOptions,
        run_configurations: analysis.MultiParameterConfiguration,
        use_public_partitions: bool,
        contribution_histograms: histograms.DatasetHistograms):
    assert len(utility_analysis_result) == run_configurations.size
    # TODO(dvadym): implement relative error.
    # TODO(dvadym): take into consideration partition selection from private
    # partition selection.
    assert tune_options.function_to_minimize == MinimizingFunction.ABSOLUTE_ERROR

    metrics = tune_options.aggregate_params.metrics[0]
    if metrics == pipeline_dp.Metrics.COUNT:
        rmse = [
            ae.count_metrics.absolute_rmse() for ae in utility_analysis_result
        ]
    else:
        rmse = [
            ae.privacy_id_count_metrics.absolute_rmse()
            for ae in utility_analysis_result
        ]
    index_best = np.argmin(rmse)

    return TuneResult(tune_options, contribution_histograms, run_configurations,
                      index_best, utility_analysis_result)


def tune(col,
         backend: pipeline_backend.PipelineBackend,
         contribution_histograms: histograms.DatasetHistograms,
         options: TuneOptions,
         data_extractors: Union[pipeline_dp.DataExtractors,
                                analysis.PreAggregateExtractors],
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

    candidates = _find_candidate_parameters(contribution_histograms,
                                            options.parameters_to_tune,
                                            options.aggregate_params.metrics[0])

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
