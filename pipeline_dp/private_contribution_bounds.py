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
"""Computation of contribution bounds in a differentially private way."""

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import pipeline_dp
from pipeline_dp import dp_computations

from pipeline_dp.histograms import Histogram
from pipeline_dp import pipeline_composite_functions as composite_funcs


class PrivateL0Calculator:
    """Calculates differentially-private l0 bound (i.e.
    max_partitions_contributed)."""

    def __init__(self,
                 params: pipeline_dp.CalculatePrivateContributionBoundsParams,
                 partitions, histograms, backend) -> None:
        """
        Args:
          params: calculation parameters.
          partitions: pCollection of all partitions present in the data
            the calculated bound will be used for.
          histograms: pCollection consisting of one element of DatasetHistograms
            object.
          backend: pipeline backend to use for calculations.
        """
        self._params = params
        self._backend = backend
        self._partitions = partitions
        self._histograms = histograms

    @dataclass
    class Inputs:
        l0_histogram: Histogram
        number_of_partitions: int
        possible_contribution_bounds: List[int]

    @lru_cache(maxsize=None)
    def calculate(self):
        """Chooses l0 contribution bound in a differentially private way.

        Returns a one element pCollection containing the value for the l0
        bound."""
        l0_histogram = self._backend.to_multi_transformable_collection(
            self._backend.map(
                self._histograms, lambda h: h.l0_contributions_histogram,
                "Extract l0_contributions_histogram from DatasetHistograms"))
        number_of_partitions = self._calculate_number_of_partitions()
        possible_contribution_bounds = self._lower_bounds_of_bins(l0_histogram)

        l0_calculation_input_col = self._backend.to_multi_transformable_collection(
            composite_funcs.collect_to_container(
                self._backend, {
                    "l0_histogram": l0_histogram,
                    "number_of_partitions": number_of_partitions,
                    "possible_contribution_bounds": possible_contribution_bounds
                }, PrivateL0Calculator.Inputs,
                "Collecting L0 calculation inputs into one object"))
        return self._backend.map(l0_calculation_input_col,
                                 lambda inputs: self._calculate_l0(inputs),
                                 "Calculate private l0 bound")

    def _calculate_l0(self, inputs: Inputs):
        scoring_function = L0ScoringFunction(self._params,
                                             inputs.number_of_partitions,
                                             inputs.l0_histogram)
        return dp_computations.ExponentialMechanism(scoring_function).apply(
            self._params.calculation_eps, inputs.possible_contribution_bounds)

    def _calculate_number_of_partitions(self):
        distinct_partitions = self._backend.distinct(
            self._partitions, "Keep only distinct partitions")
        return composite_funcs.size(self._backend, distinct_partitions,
                                    "Calculate number of partitions")

    def _lower_bounds_of_bins(self, histogram_col):
        # TODO: is correct to do so? does it preserve dp? The problem is that
        # we infer it from the histogram that used raw data and if for example
        # there are 7 different public partitions but each user contributed to 6
        # partitions max, then we will choose only from 1..6 and not 1..7, but
        # it seems incorrect because we eliminate one of the options by
        # observing the data. (Example taken from restaurant visits).

        def histogram_to_bin_lowers(hist: Histogram) -> List[int]:
            return list(map(lambda bin: bin.lower, hist.bins))

        return self._backend.map(histogram_col, histogram_to_bin_lowers,
                                 "Extract lowers of bins from histogram")


class L0ScoringFunction(dp_computations.ExponentialMechanism.ScoringFunction):
    """Function to score different max_partitions_contributed bounds.

    Suitable only for COUNT and PRIVACY_ID_COUNT aggregations."""

    def __init__(self,
                 params: pipeline_dp.CalculatePrivateContributionBoundsParams,
                 number_of_partitions: int, l0_histogram: Histogram):
        super().__init__()
        self._params = params
        self._number_of_partitions = number_of_partitions
        self._l0_histogram = l0_histogram

    def score(self, k: int) -> float:
        """
        P := number_of_partitions
        std := count_noise_std
        B := _max_partitions_contributed_best_upper_bound (= global_sensitivity)
        D := dataset
        uid := user identifier
        #contributions(uid, D) = number of partitions in D where uid contributed
        at least once.

        score(k) = -0.5 * impact_noise(k) - 0.5 * impact_dropped(k) =(1)
        -0.5 * P * std - 0.5 * Σ_uid max(min(#contributions(uid, D), B) - k, 0)

        Note: linf = 1, because impact_noise and impact_dropped are proportional
        to linf, and we can omit it in (1) equality.

        k is l0 for which std is calculated.
        Laplace noise:
        std = sqrt(2 * (l0 / ε)^2) = k / ε * sqrt(2)
        Gaussian noise:
        std is calculated based on https://arxiv.org/abs/1805.06530v2.
        """
        impact_noise_weight = 0.5
        return -(impact_noise_weight * self._l0_impact_noise(k) +
                 (1 - impact_noise_weight) * self._l0_impact_dropped(k))

    def _max_partitions_contributed_best_upper_bound(self):
        return min(self._params.max_partitions_contributed_upper_bound,
                   self._number_of_partitions)

    @property
    def global_sensitivity(self) -> float:
        """Global sensitivity of the scoring function.

        Equals min(l0_upper_bound, number_of_partitions), because
        max_partitions_contributed upper bound is always at least
        number of partitions."""
        return self._max_partitions_contributed_best_upper_bound()

    @property
    def is_monotonic(self) -> bool:
        """score(k) for l0 is monotonic."""
        return True

    def _l0_impact_noise(self, k: int):
        noise_params = dp_computations.ScalarNoiseParams(
            eps=self._params.aggregation_eps,
            delta=self._params.aggregation_delta,
            max_partitions_contributed=k,
            max_contributions_per_partition=1,
            noise_kind=self._params.aggregation_noise_kind,
            min_value=None,
            max_value=None,
            min_sum_per_partition=None,
            max_sum_per_partition=None)
        return (self._number_of_partitions *
                dp_computations.compute_dp_count_noise_std(noise_params))

    def _l0_impact_dropped(self, k: int):
        # TODO: precalculate it and make it work in O(1) time.
        capped_contributions = map(
            lambda bin: max(
                min(
                    bin.lower,
                    self._max_partitions_contributed_best_upper_bound(),
                ) - k,
                0,
            ) * bin.count,
            self._l0_histogram.bins,
        )
        return sum(capped_contributions)
