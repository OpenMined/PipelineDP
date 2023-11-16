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
import logging
import math
from numbers import Number

import pipeline_dp
from pipeline_dp import pipeline_backend
from pipeline_dp import input_validators
from pipeline_dp.dataset_histograms import histograms
import analysis
from analysis import metrics
from analysis import utility_analysis
from analysis import dp_strategy_selector

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union, Sequence
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
    noise_kind: pipeline_dp.NoiseKind = True

    # Partition selection strategy is tuned always.

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
        parameters_to_tune: ParametersToTune,
        metric: Optional[pipeline_dp.Metric],
        max_candidates: int) -> analysis.MultiParameterConfiguration:
    """Finds candidates for l0, l_inf and max_sum_per_partition_bounds parameters.

    Args:
        hist: dataset contribution histogram.
        parameters_to_tune: which parameters to tune.
        metric: dp aggregation for which candidates are computed. If metric is
          None, it means no metrics to compute, i.e. only select partitions.
        max_candidates: how many candidates ((l0, linf) pairs) can be in the
          output. Note that output can contain fewer candidates. 100 is default
          heuristically chosen value, better to adjust it for your use-case.
    """
    calculate_l0_param = parameters_to_tune.max_partitions_contributed
    generate_linf_count = metric == pipeline_dp.Metrics.COUNT
    generate_max_sum_per_partition = metric == pipeline_dp.Metrics.SUM
    calculate_linf_count = (parameters_to_tune.max_contributions_per_partition
                            and generate_linf_count)
    calculate_sum_per_partition_param = (
        parameters_to_tune.max_sum_per_partition and
        generate_max_sum_per_partition)
    l0_bounds = linf_bounds = None
    max_sum_per_partition_bounds = min_sum_per_partition_bounds = None

    if calculate_sum_per_partition_param:
        if hist.linf_sum_contributions_histogram.bins[0].lower < 0:
            logging.warning(
                "max_sum_per_partition should not contain negative sums because"
                " min_sum_per_partition tuning is not supported yet and "
                "therefore tuning for max_sum_per_partition works only when "
                "linf_sum_contributions_histogram does not negative sums")

    if calculate_l0_param and calculate_linf_count:
        l0_bounds, linf_bounds = _find_candidates_parameters_in_2d_grid(
            hist.l0_contributions_histogram, hist.linf_contributions_histogram,
            _find_candidates_constant_relative_step,
            _find_candidates_constant_relative_step, max_candidates)
    elif calculate_l0_param and calculate_sum_per_partition_param:
        l0_bounds, max_sum_per_partition_bounds = _find_candidates_parameters_in_2d_grid(
            hist.l0_contributions_histogram,
            hist.linf_sum_contributions_histogram,
            _find_candidates_constant_relative_step,
            _find_candidates_bins_max_values_subsample, max_candidates)
        min_sum_per_partition_bounds = [0] * len(max_sum_per_partition_bounds)
    elif calculate_l0_param:
        l0_bounds = _find_candidates_constant_relative_step(
            hist.l0_contributions_histogram, max_candidates)
    elif calculate_linf_count:
        linf_bounds = _find_candidates_constant_relative_step(
            hist.linf_contributions_histogram, max_candidates)
    elif calculate_sum_per_partition_param:
        max_sum_per_partition_bounds = _find_candidates_bins_max_values_subsample(
            hist.linf_sum_contributions_histogram, max_candidates)
        min_sum_per_partition_bounds = [0] * len(max_sum_per_partition_bounds)
    else:
        assert False, "Nothing to tune."

    return analysis.MultiParameterConfiguration(
        max_partitions_contributed=l0_bounds,
        max_contributions_per_partition=linf_bounds,
        min_sum_per_partition=min_sum_per_partition_bounds,
        max_sum_per_partition=max_sum_per_partition_bounds)


def _add_dp_strategy_to_multi_parameter_configuration(
        configuration: analysis.MultiParameterConfiguration,
        noise_kind: Optional[pipeline_dp.NoiseKind],
        strategy_selector: dp_strategy_selector.DPStrategySelector) -> None:
    if noise_kind is not None and strategy_selector.is_public_partitions:
        return
    params = [configuration.get_aggregate_params(i) for i in configuration.size]
    # Initialize fields corresponding to DP strategy configuration
    if noise_kind is None:
        configuration.noise_kind = []
    if not strategy_selector.is_public_partitions:
        configuration.partition_selection_strategy = []
    for param in params:
        pass


def _find_candidates_parameters_in_2d_grid(
        hist1: histograms.Histogram, hist2: histograms.Histogram,
        find_candidates_func1: Callable[[histograms.Histogram, int],
                                        Sequence[Number]],
        find_candidates_func2: Callable[[histograms.Histogram, int],
                                        Sequence[Number]],
        max_candidates: int) -> Tuple[Sequence[Number], Sequence[Number]]:
    """Finds candidates for 2 parameters.

    If we have 2 parameters to tune, then candidates for them form a 2
    dimensional grid. If for one parameter there is less than
    sqrt(max_candidates) candidates, we can add more candidates for the other
    parameter. This function implements this logic.

    Args:
        hist1: histogram of the distribution of the first parameter.
        hist2: histogram of the distribution of the second parameter.
        find_candidates_func1: function that given hist1 and maximum of
          candidates finds the candidates.
        find_candidates_func2: function that given hist2 and maximum of
          candidates finds the candidates.
        max_candidates: maximum number of the candidates to produce.
    Returns:
        Two sequences which represent pairs of candidates for parameters 1 and
          2. Sequences are of the same length and their lengths do not exceed
          max_candidates.
    """

    max_candidates_per_parameter = int(math.sqrt(max_candidates))
    param1_candidates = find_candidates_func1(hist1,
                                              max_candidates_per_parameter)
    param2_candidates = find_candidates_func2(hist2,
                                              max_candidates_per_parameter)
    param1_bounds, param2_bounds = [], []

    # if param1 or param2 has fewer candidates than requested then we can add
    # more candidates for the other parameter.
    if (len(param2_candidates) < max_candidates_per_parameter and
            len(param1_candidates) == max_candidates_per_parameter):
        param1_candidates = find_candidates_func1(
            hist1, int(max_candidates / len(param2_candidates)))
    elif (len(param1_candidates) < max_candidates_per_parameter and
          len(param2_candidates) == max_candidates_per_parameter):
        param2_candidates = find_candidates_func2(
            hist2, int(max_candidates / len(param1_candidates)))

    for param1 in param1_candidates:
        for param2 in param2_candidates:
            param1_bounds.append(param1)
            param2_bounds.append(param2)

    return param1_bounds, param2_bounds


def _find_candidates_constant_relative_step(histogram: histograms.Histogram,
                                            max_candidates: int) -> List[int]:
    """Finds candidates with constant relative step.

    Candidates are a sequence starting from 1 where relative difference
    between two neighbouring elements is the same. Mathematically it means
    that candidates are a sequence a_i, where
    a_i = max_value^(i / (max_candidates - 1)), i in [0..(max_candidates - 1)]
    """
    max_value = histogram.max_value()
    assert max_value >= 1, "max_value has to be >= 1."
    max_candidates = min(max_candidates, max_value)
    assert max_candidates > 0, "max_candidates have to be positive"
    if max_candidates == 1:
        return [1]
    step = pow(max_value, 1 / (max_candidates - 1))
    candidates = [1]
    accumulated = 1
    for i in range(1, max_candidates):
        previous_candidate = candidates[-1]
        if previous_candidate >= max_value:
            break
        accumulated *= step
        next_candidate = max(previous_candidate + 1, math.ceil(accumulated))
        candidates.append(next_candidate)
    # float calculations might be not precise enough but the last candidate has
    # to be always max_value
    candidates[-1] = max_value
    return candidates


def _find_candidates_bins_max_values_subsample(
        histogram: histograms.Histogram, max_candidates: int) -> List[float]:
    """Takes max values of histogram bins with constant step between each other."""
    max_candidates = min(max_candidates, len(histogram.bins))
    ids = np.round(np.linspace(0, len(histogram.bins) - 1,
                               num=max_candidates)).astype(int)
    bin_maximums = np.fromiter(map(lambda bin: bin.max, histogram.bins),
                               dtype=float)
    return bin_maximums[ids].tolist()


def tune(col,
         backend: pipeline_backend.PipelineBackend,
         contribution_histograms: histograms.DatasetHistograms,
         options: TuneOptions,
         data_extractors: Union[pipeline_dp.DataExtractors,
                                pipeline_dp.PreAggregateExtractors],
         public_partitions=None,
         strategy_selector_factory: dp_strategy_selector.
         DPStrategySelectorFactory = None):
    """Tunes parameters.

    It works in the following way:
        1. Find candidates for contribution bounding parameters.
        2. Utility analysis run for those parameters.
        3. The best parameter set is chosen according to
          options.minimizing_function.

    The result contains output metrics for all utility analysis which were
    performed.

    For tuning parameters for DPEngine.select_partitions set
      options.aggregate_params.metrics to an empty list.

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
        strategy_selector_factory: todo
    Returns:
       returns tuple (1 element collection which contains TuneResult,
        a collection which contains utility analysis results per partition).
    """
    _check_tune_args(options, public_partitions is not None)
    if strategy_selector_factory is None:
        strategy_selector_factory = dp_strategy_selector.DPStrategySelectorFactory(
        )

    metric = None
    if options.aggregate_params.metrics:
        metric = options.aggregate_params.metrics[0]

    candidates: analysis.MultiParameterConfiguration = _find_candidate_parameters(
        contribution_histograms, options.parameters_to_tune, metric,
        options.number_of_parameter_candidates)

    # Add DP strategy (noise_kind, partition_selection_strategy) to multi
    # parameter configuration.
    noise_kind = None
    if not options.parameters_to_tune.noise_kind:
        noise_kind = options.aggregate_params.noise_kind
    strategy_selector = strategy_selector_factory.create(
        options.epsilon,
        options.delta,
        metric,
        is_public_partitions=public_partitions is not None)
    _add_dp_strategy_to_multi_parameter_configuration(candidates, noise_kind,
                                                      strategy_selector)

    utility_analysis_options = analysis.UtilityAnalysisOptions(
        epsilon=options.epsilon,
        delta=options.delta,
        aggregate_params=options.aggregate_params,
        multi_param_configuration=candidates,
        partitions_sampling_prob=options.partitions_sampling_prob,
        pre_aggregated_data=options.pre_aggregated_data)

    utility_result, per_partition_utility_result = utility_analysis.perform_utility_analysis(
        col, backend, utility_analysis_options, data_extractors,
        public_partitions)
    # utility_result: (UtilityReport)
    # per_partition_utility_result: (pk, (PerPartitionMetrics))
    use_public_partitions = public_partitions is not None

    utility_result = backend.to_list(utility_result, "To list")
    # 1 element collection with list[UtilityReport]
    utility_result = backend.map(
        utility_result, lambda result: _convert_utility_analysis_to_tune_result(
            result, options, candidates, use_public_partitions,
            contribution_histograms), "To Tune result")
    return utility_result, per_partition_utility_result


def _convert_utility_analysis_to_tune_result(
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
    sorted_utility_reports = sorted(utility_reports,
                                    key=lambda e: e.configuration_index)

    index_best = -1  # not found
    # Find best index if there are metrics to compute. Absence of metrics to
    # compute means that this is SelectPartition analysis.
    if tune_options.aggregate_params.metrics:
        rmse = [
            ur.metric_errors[0].absolute_error.rmse
            for ur in sorted_utility_reports
        ]
        index_best = np.argmin(rmse)

    return TuneResult(tune_options,
                      contribution_histograms,
                      run_configurations,
                      index_best,
                      utility_reports=sorted_utility_reports)


def _check_tune_args(options: TuneOptions, is_public_partitions: bool):
    # Check metrics to tune.
    metrics = options.aggregate_params.metrics
    if not metrics:
        # Empty metrics means tuning for select_partitions.
        if is_public_partitions:
            # Empty metrics means that partition selection tuning is performed.
            raise ValueError("Empty metrics means tuning of partition selection"
                             " but public partitions were provided.")
    elif len(metrics) > 1:
        raise ValueError(
            f"Tuning supports only one metric, but {metrics} given.")
    else:  # len(metrics) == 1
        if metrics[0] not in [
                pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT,
                pipeline_dp.Metrics.SUM
        ]:
            raise ValueError(
                f"Tuning is supported only for Count, Privacy id count and Sum, but {metrics[0]} given."
            )

    if options.parameters_to_tune.min_sum_per_partition:
        raise ValueError(
            "Tuning of min_sum_per_partition is not supported yet.")

    if options.function_to_minimize != MinimizingFunction.ABSOLUTE_ERROR:
        raise NotImplementedError(
            f"Only {MinimizingFunction.ABSOLUTE_ERROR} is implemented.")
