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
        # TODO: add values for mean and variance
        raise ValueError("Invalid metric")


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
