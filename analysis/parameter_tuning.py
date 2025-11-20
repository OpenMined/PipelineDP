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
from pipeline_dp.aggregate_params import NoiseKind, Metrics, AggregateParams
from pipeline_dp.data_extractors import PreAggregateExtractors, DataExtractors
from pipeline_dp import dp_computations
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
    noise_kind: NoiseKind = True

    # Partition selection strategy is tuned always.

    def __post_init__(self):
        if not any(dataclasses.asdict(self).values()):
            raise ValueError("ParametersToTune must have at least 1 parameter "
                             "to tune.")


@dataclass
class TuneOptions:
    """Options for the tuning process.

    Note that parameters that are not tuned (e.g., metrics, noise kind) are
    taken
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
    aggregate_params: AggregateParams
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
         (best) configuration in utility_analysis_parameters. Note that those
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
        parameters_to_tune: ParametersToTune, aggregate_params: AggregateParams,
        max_candidates: int) -> analysis.MultiParameterConfiguration:
    """Finds candidates for l0, l_inf and max_sum_per_partition_bounds
    parameters.

    Args:
        aggregate_params: parameters of aggregation.
        hist: dataset contribution histogram.
        parameters_to_tune: which parameters to tune.
        max_candidates: how many candidates ((l0, linf) pairs) can be in the
          output. Note that output can contain fewer candidates. 100 is the
          default
          heuristically chosen value, better to adjust it for your use-case.
    """
    # L0 bounds
    tune_l0 = parameters_to_tune.max_partitions_contributed
    metrics = aggregate_params.metrics
    tune_count_linf = (parameters_to_tune.max_contributions_per_partition and
                       Metrics.COUNT in metrics)
    tune_sum_linf = (parameters_to_tune.max_sum_per_partition and
                     Metrics.SUM in metrics)

    # Find L0 candidates (i.e. max_partitions_contributed)
    if tune_l0:
        if tune_count_linf or tune_sum_linf:
            max_l0_candidates = int(np.sqrt(max_candidates))
        else:
            max_l0_candidates = max_candidates
        l0_bounds = _find_candidates_constant_relative_step(
            hist.l0_contributions_histogram, max_l0_candidates)
    else:  # no l0 tuning
        l0 = aggregate_params.max_partitions_contributed
        l0_bounds = [l0] if l0 else [1]

    max_linf_candidates = max_candidates // len(l0_bounds)

    # Find Linf count candidates (i.e. max_contributions_per_partition)
    linf_count_bounds = None
    if tune_count_linf:
        linf_count_bounds = _find_candidates_constant_relative_step(
            hist.linf_contributions_histogram, max_linf_candidates)
    elif Metrics.COUNT in metrics:
        linf_count_bounds = [aggregate_params.max_contributions_per_partition]

    linf_sum_bounds = None
    if tune_sum_linf:
        n_sum_columns = hist.num_sum_histograms()
        linf_sum_bounds = []
        if n_sum_columns == 1:
            linf_sum_bounds.append(
                _find_candidates_bins_max_values_subsample(
                    hist.linf_sum_contributions_histogram, max_linf_candidates))
        else:  # n_sums > 1
            for i in range(n_sum_columns):
                linf_sum_bounds.append(
                    _find_candidates_bins_max_values_subsample(
                        hist.linf_sum_contributions_histogram[i],
                        max_linf_candidates))

    # Linf COUNT and SUM bounds can have different number of elements, for
    # running Utility Analysis it is required that they have the same length.
    # Let us pad each to the max_length with 0-th element.
    max_linf_len = 1  # max len of counts and sum bounds
    if tune_count_linf:
        max_linf_len = len(linf_count_bounds)
    if tune_sum_linf:
        max_linf_len = max(max_linf_len, max(map(len, linf_sum_bounds)))

    if tune_count_linf:
        _pad_list(linf_count_bounds, max_linf_len)
    if tune_sum_linf:
        for a in linf_sum_bounds:
            _pad_list(a, max_linf_len)

    min_sum_per_partition = max_sum_per_partition = None
    if tune_sum_linf:
        max_linf_len = max(max_linf_len, len(linf_sum_bounds))
        n_sum_columns = hist.num_sum_histograms()
        if n_sum_columns == 1:
            max_sum_per_partition = linf_sum_bounds[0]
            min_sum_per_partition = [0] * len(max_sum_per_partition)
        else:  # n_sum_columns > 1
            max_sum_per_partition = list(zip(*linf_sum_bounds))
            min_sum_per_partition = [
                (0,) * n_sum_columns for _ in range(len(max_sum_per_partition))
            ]

    # Make cross-product of l0 and linf bounds. That is done by duplicating
    # each element of l0 bounds and by duplicating the whole arrays of linf
    # bounds. Example if l0_bound = [1,2] and linf_bounds = [3,4], then
    # l0_bounds will be [1,1,2,2] and linf_bounds will be [3,4,3,4].
    l0_duplication = max_linf_len
    linf_duplication = 1 if l0_bounds is None else len(l0_bounds)
    return analysis.MultiParameterConfiguration(
        max_partitions_contributed=_duplicate_each_element(
            l0_bounds, l0_duplication),
        max_contributions_per_partition=_duplicate_list(linf_count_bounds,
                                                        linf_duplication),
        min_sum_per_partition=_duplicate_list(min_sum_per_partition,
                                              linf_duplication),
        max_sum_per_partition=_duplicate_list(max_sum_per_partition,
                                              linf_duplication),
    )


def _pad_list(a: Optional[List], size: int):
    if a is not None and len(a) < size:
        a.extend([a[0]] * (size - len(a)))


def _duplicate_each_element(a: Optional[List], n: int) -> Optional[List]:
    if a is None:
        return None
    return [x for x in a for _ in range(n)]


def _duplicate_list(a: Optional[List], n: int) -> Optional[List]:
    if a is None:
        return None
    return a * n


def _add_dp_strategy_to_multi_parameter_configuration(
        configuration: analysis.MultiParameterConfiguration,
        blueprint_params: AggregateParams, noise_kind: Optional[NoiseKind],
        strategy_selector: dp_strategy_selector.DPStrategySelector) -> None:
    params = [
        # get_aggregate_params returns a tuple (AggregateParams,
        # min_max_sum_per_partitions)
        # for multi-columns. DP Strategy (i.e. noise_kind, partition_selection)
        # is independent from min_max_sum_per_partitions, it's fine to just get
        # the first element of AggregateParam
        configuration.get_aggregate_params(blueprint_params, i)[0]
        for i in range(configuration.size)
    ]
    # Initialize fields corresponding to DP strategy configuration
    configuration.noise_kind = []
    find_partition_selection = not strategy_selector.is_public_partitions
    if find_partition_selection:
        configuration.partition_selection_strategy = []
    for param in params:
        # linf sensitivity does not influence strategy choosing, so it's ok
        # to set linf=1.
        sensitivities = dp_computations.Sensitivities(
            l0=param.max_partitions_contributed, linf=1)
        dp_strategy = strategy_selector.get_dp_strategy(sensitivities)
        if noise_kind is None:
            configuration.noise_kind.append(dp_strategy.noise_kind)
        else:
            configuration.noise_kind.append(noise_kind)
        if find_partition_selection:
            configuration.partition_selection_strategy.append(
                dp_strategy.partition_selection_strategy)


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
    """Takes max values of histogram bins with constant step between each
    other."""
    # In order to ensure that max_sum_per_partition > 0, let us skip 0-th
    # bin if max = 0.
    # TODO(dvadym): better algorithm for finding candidates.
    from_bin_idx = 0 if histogram.bins[0].max > 0 else 1
    max_candidates = min(max_candidates, len(histogram.bins) - from_bin_idx)
    ids = np.round(
        np.linspace(from_bin_idx, len(histogram.bins) - 1,
                    num=max_candidates)).astype(int)
    bin_maximums = np.fromiter(map(lambda bin: bin.max, histogram.bins),
                               dtype=float)
    return bin_maximums[ids].tolist()


def tune(col,
         backend: pipeline_backend.PipelineBackend,
         contribution_histograms: histograms.DatasetHistograms,
         options: TuneOptions,
         data_extractors: Union[DataExtractors, PreAggregateExtractors],
         public_partitions=None,
         strategy_selector_factory: Optional[
             dp_strategy_selector.DPStrategySelectorFactory] = None,
         candidates: Optional[analysis.MultiParameterConfiguration] = None):
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
        strategy_selector_factory: factory for creating StrategySelector. If
          non provided, the DPStrategySelector will be used.
    Returns:
       returns tuple (1 element collection which contains TuneResult,
        a collection which contains utility analysis results per partition).
    """
    _check_tune_args(options, public_partitions is not None)
    if strategy_selector_factory is None:
        strategy_selector_factory = dp_strategy_selector.DPStrategySelectorFactory(
        )

    if candidates is None:
        candidates: analysis.MultiParameterConfiguration = (
            _find_candidate_parameters(
                hist=contribution_histograms,
                parameters_to_tune=options.parameters_to_tune,
                aggregate_params=options.aggregate_params,
                max_candidates=options.number_of_parameter_candidates,
            ))

    # Add DP strategy (noise_kind, partition_selection_strategy) to multi
    # parameter configuration.
    noise_kind = None
    if not options.parameters_to_tune.noise_kind:
        noise_kind = options.aggregate_params.noise_kind

    strategy_selector = strategy_selector_factory.create(
        options.epsilon,
        options.delta,
        metrics=options.aggregate_params.metrics,
        is_public_partitions=public_partitions is not None)
    _add_dp_strategy_to_multi_parameter_configuration(candidates,
                                                      options.aggregate_params,
                                                      noise_kind,
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
    # Find the best index if there are metrics to compute. The absence of
    # metrics to compute means that this is SelectPartition analysis.
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
    else:  # len(metrics) == 1
        if metrics[0] not in [
                Metrics.COUNT, Metrics.PRIVACY_ID_COUNT, Metrics.SUM
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
