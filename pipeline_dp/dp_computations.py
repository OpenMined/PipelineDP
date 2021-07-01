"""Differential privacy computing of count, sum, mean, variance."""

import numpy as np
import pipeline_dp

from dataclasses import dataclass


@dataclass
class MeanVarParams:
    """The parameters used for computing the dp sum, count, mean, variance."""
    eps: float
    delta: float
    low: float
    high: float
    max_partitions_contributed: int
    max_contributions_per_partition: int
    noise_kind: pipeline_dp.NoiseKind  # Laplace or Gaussian

    def l0_sensitivity(self):
        return self.max_partitions_contributed

    def linf_sensitivity(self, metric):
        if metric == pipeline_dp.Metrics.COUNT:
            return self.max_contributions_per_partition
        if metric == pipeline_dp.Metrics.SUM:
            return self.max_contributions_per_partition * max(
                abs(self.low), abs(self.high))
        if metric == pipeline_dp.Metrics.MEAN:
            return self.max_contributions_per_partition * abs(
                self.high - self.low) / 2
        # TODO: add values for mean and variance
        raise ValueError("Invalid metric")

    def middle(self):
        return self.low + (self.high - self.low) / 2


def compute_l1_sensitivity(l0_sensitivity: float, linf_sensitivity: float):
    return l0_sensitivity * linf_sensitivity


def compute_l2_sensitivity(l0_sensitivity: float, linf_sensitivity: float):
    return np.sqrt(l0_sensitivity) * linf_sensitivity


def compute_sigma(eps: float, delta: float, l2_sensitivity: float):
    # TODO: use the optimal sigma.
    # Theorem 3.22: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    return np.sqrt(2 * np.log(1.25 / delta)) * l2_sensitivity / eps


def apply_laplace_mechanism(value: float, eps: float, l1_sensitivity: float):
    # TODO: use the secure noise instead of np.random
    return value + np.random.laplace(0, l1_sensitivity / eps)


def apply_gaussian_mechanism(value: float, eps: float, delta: float,
                             l2_sensitivity: float):
    sigma = compute_sigma(eps, delta, l2_sensitivity)
    # TODO: use the secure noise instead of np.random
    return value + np.random.normal(0, sigma)


def _add_random_noise(value: float, eps: float, delta: float,
                      l0_sensitivity: float, linf_sensitivity: float,
                      noise_kind: pipeline_dp.NoiseKind):
    if noise_kind == pipeline_dp.NoiseKind.LAPLACE:
        l1_sensitivity = compute_l1_sensitivity(l0_sensitivity,
                                                linf_sensitivity)
        return apply_laplace_mechanism(value, eps, l1_sensitivity)
    if noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
        l2_sensitivity = compute_l2_sensitivity(l0_sensitivity,
                                                linf_sensitivity)
        return apply_gaussian_mechanism(value, eps, delta, l2_sensitivity)
    raise ValueError("Noise kind must be either Laplace or Gaussian.")


def equally_split_budget(eps: float, delta: float, no_mechanisms: int):
    """Equally splits the budget (eps, delta) between a given number of mechanisms.

    Args:
        eps, delta: Parameters of (epsilon, delta)-differential privacy.
        no_mechanisms: The number of mechanisms between which we split the budget.

    Raises:
        ValueError: The number of mechanisms must be a natural non-zero number.

    Returns:
        An array with the split budgets.
    """
    if no_mechanisms <= 0:
        raise ValueError(
            "The number of mechanisms must be a natural non-zero number.")

    budgets = [(eps / no_mechanisms, delta / no_mechanisms) for _ in
               range(no_mechanisms - 1)]
    last_mechanism_budget = (eps - (no_mechanisms - 1) * (eps / no_mechanisms),
                             delta - (no_mechanisms - 1) * (
                                     delta / no_mechanisms))
    budgets.append(last_mechanism_budget)
    return budgets


def compute_dp_count(count: int, dp_params: MeanVarParams):
    """Computes DP count.

    Args:
        count: Non-DP count.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.
    """
    l0_sensitivity = dp_params.l0_sensitivity()
    linf_sensitivity = dp_params.linf_sensitivity(pipeline_dp.Metrics.COUNT)

    return _add_random_noise(count, dp_params.eps, dp_params.delta,
                             l0_sensitivity, linf_sensitivity,
                             dp_params.noise_kind)


def compute_dp_sum(sum: float, dp_params: MeanVarParams):
    """Computes DP sum.

    Args:
        sum: Non-DP sum.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.
    """
    l0_sensitivity = dp_params.l0_sensitivity()
    linf_sensitivity = dp_params.linf_sensitivity(pipeline_dp.Metrics.SUM)

    return _add_random_noise(sum, dp_params.eps, dp_params.delta,
                             l0_sensitivity, linf_sensitivity,
                             dp_params.noise_kind)


def compute_dp_mean(count: int, sum: float, dp_params: MeanVarParams):
    """Computes DP mean.

    Args:
        count: Non-DP count.
        sum: Non-DP sum.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.
    """
    middle = dp_params.middle()
    normalized_sum = sum - count * middle

    l0_sensitivity = dp_params.l0_sensitivity()

    # Splits the budget equally between the two mechanisms.
    (sum_eps, sum_delta), (count_eps, count_delta) = equally_split_budget(
        dp_params.eps, dp_params.delta, 2)

    dp_normalized_sum = _add_random_noise(normalized_sum, sum_eps, sum_delta,
                                          l0_sensitivity,
                                          dp_params.linf_sensitivity(
                                              pipeline_dp.Metrics.MEAN),
                                          dp_params.noise_kind)
    dp_count = _add_random_noise(count, count_eps, count_delta, l0_sensitivity,
                                 dp_params.linf_sensitivity(
                                     pipeline_dp.Metrics.COUNT),
                                 dp_params.noise_kind)
    dp_mean = dp_normalized_sum / dp_count + middle
    return dp_count, dp_normalized_sum + count * middle, dp_mean
