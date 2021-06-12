"""DP computing of count, sum, mean, variance."""

import numpy as np
import pipeline_dp

from dataclasses import dataclass


@dataclass
class MeanVarParams:
    """Manages the parameters used for computing the mean/variance."""
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
        return self.max_contributions_per_partition * {
            pipeline_dp.Metrics.COUNT: 1,
            pipeline_dp.Metrics.SUM: np.max(self.low, self.high)
        }.get(metric)


def _l1_sensitivity(l0_sensitivity, linf_sensitivity):
    return l0_sensitivity * linf_sensitivity


def _l2_sensitivity(l0_sensitivity, linf_sensitivity):
    return np.sqrt(l0_sensitivity) * linf_sensitivity


def _compute_sigma(eps, delta, l2_sensitivity):
    # Theorem 3.22: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    return np.sqrt(2 * np.log(1.25 / delta)) * l2_sensitivity / eps


def _apply_laplace_mechanism(value, eps, l1_sensitivity):
    # TODO: use the secure noise instead of np.random
    return value + np.random.laplace(0, l1_sensitivity / eps)


def _apply_gaussian_mechanism(value, eps, delta, l2_sensitivity):
    sigma = _compute_sigma(eps, delta, l2_sensitivity)

    # TODO: use the secure noise instead of np.random
    return value + np.random.normal(0, np.power(sigma, 2) * (eps, delta, l2_sensitivity))


def _add_random_noise(value, eps, delta, l0_sensitivity, linf_sensitivity, noise_kind):
    if noise_kind == pipeline_dp.NoiseKind.LAPLACE:
        l1_sensitivity = _l1_sensitivity(l0_sensitivity, linf_sensitivity)
        return _apply_laplace_mechanism(value, eps, l1_sensitivity)
    if noise_kind == pipeline_dp.NoiseKind.GAUSSIAN:
        l2_sensitivity = _l2_sensitivity(l0_sensitivity, linf_sensitivity)
        return _apply_gaussian_mechanism(value, eps, delta, l2_sensitivity)
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

    return _add_random_noise(count, dp_params.eps, dp_params.delta, l0_sensitivity, linf_sensitivity,
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

    return _add_random_noise(sum, dp_params.eps, dp_params.delta, l0_sensitivity, linf_sensitivity,
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
    dp_count = compute_dp_count(count, dp_params)
    dp_sum = compute_dp_sum(sum, dp_params)
    return dp_count, dp_sum, dp_sum / dp_count


def compute_dp_var(count: int, sum: float, sum_squares: float, dp_params: MeanVarParams):
    """Computes DP variance.

    Args:
        count: Non-DP count.
        sum: Non-DP sum.
        sum_squares: Non-DP sum of squares.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.
    """
    dp_count = compute_dp_count(count, dp_params)
    dp_sum = compute_dp_sum(sum, dp_params)
    dp_sum_squares = compute_dp_sum(sum_squares, dp_params)
    dp_mean = compute_dp_mean(count, sum, dp_params)
    return dp_count, dp_sum, dp_mean, dp_sum_squares / dp_count - np.power(dp_mean, 2)
