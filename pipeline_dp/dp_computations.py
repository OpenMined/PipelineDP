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
        """"Returns the L0 sensitivity of the parameters."""
        return self.max_partitions_contributed

    def squares_interval(self):
        """Returns the bounds of the interval [low^2, high^2]."""
        if self.low < 0 and self.high > 0:
            return 0, max(self.low**2, self.high**2)
        return self.low**2, self.high**2


def compute_middle(low: float, high: float):
    """"Returns the middle point of the interval [low, high]."""
    return low + (high - low) / 2


def compute_l1_sensitivity(l0_sensitivity: float, linf_sensitivity: float):
    """Calculates the L1 sensitivity based on the L0 and Linf sensitivities.

    Args:
        l0_sensitivity: The L0 sensitivity.
        linf_sensitivity: The Linf sensitivity.

    Returns:
        The L1 sensitivity.
    """
    return l0_sensitivity * linf_sensitivity


def compute_l2_sensitivity(l0_sensitivity: float, linf_sensitivity: float):
    """Calculates the L2 sensitivity based on the L0 and Linf sensitivities.

    Args:
        l0_sensitivity: The L0 sensitivity.
        linf_sensitivity: The Linf sensitivity.

    Returns:
        The L2 sensitivity.
    """
    return np.sqrt(l0_sensitivity) * linf_sensitivity


def compute_sigma(eps: float, delta: float, l2_sensitivity: float):
    """Returns the optimal value of sigma for the Gaussian mechanism.

    Args:
        eps: The epsilon value.
        delta: The delta value.
        l2_sensitivity: The L2 sensitivity.
    """
    # TODO: use the optimal sigma.
    # Theorem 3.22: https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf
    return np.sqrt(2 * np.log(1.25 / delta)) * l2_sensitivity / eps


def apply_laplace_mechanism(value: float, eps: float, l1_sensitivity: float):
    """Applies the Laplace mechanism to the value.

    Args:
        value: The initial value.
        eps: The epsilon value.
        l1_sensitivity: The L1 sensitivity.

    Returns:
        The value resulted after adding the noise.
    """
    # TODO: use the secure noise instead of np.random
    return value + np.random.laplace(0, l1_sensitivity / eps)


def apply_gaussian_mechanism(value: float, eps: float, delta: float,
                             l2_sensitivity: float):
    """Applies the Gaussian mechanism to the value.

    Args:
        value: The initial value.
        eps: The epsilon value.
        delta: The delta value.
        l2_sensitivity: The L2 sensitivity.

    Returns:
        The value resulted after adding the noise.
    """
    sigma = compute_sigma(eps, delta, l2_sensitivity)
    # TODO: use the secure noise instead of np.random
    return value + np.random.normal(0, sigma)


def _add_random_noise(value: float, eps: float, delta: float,
                      l0_sensitivity: float, linf_sensitivity: float,
                      noise_kind: pipeline_dp.NoiseKind):
    """Adds random noise according to the parameters.

    Args:
        value: The initial value.
        eps: The epsilon value.
        delta: The delta value.
        l0_sensitivity: The L0 sensitivity.
        linf_sensitivity: The Linf sensitivity.
        noise_kind: The kind of noise used.

    Returns:
        The value resulted after adding the random noise.
    """
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
        eps, delta: The available budget.
        no_mechanisms: The number of mechanisms between which we split the budget.

    Raises:
        ValueError: The number of mechanisms must be a positive integer.

    Returns:
        An array with the split budgets.
    """
    if no_mechanisms <= 0:
        raise ValueError("The number of mechanisms must be a positive integer.")

    # These variables are used to keep track of the budget used.
    # In this way, we can improve accuracy of floating-point operations.
    eps_used = delta_used = 0
    budgets = []

    for _ in range(no_mechanisms - 1):
        budget = (eps / no_mechanisms, delta / no_mechanisms)
        eps_used += budget[0]
        delta_used += budget[1]
        budgets.append(budget)

    budgets.append((eps - eps_used, delta - delta_used))
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
    linf_sensitivity = dp_params.max_contributions_per_partition

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
    linf_sensitivity = dp_params.max_contributions_per_partition * max(
        abs(dp_params.low), abs(dp_params.high))

    return _add_random_noise(sum, dp_params.eps, dp_params.delta,
                             l0_sensitivity, linf_sensitivity,
                             dp_params.noise_kind)


def _compute_mean(count: float, dp_count: float, sum: float, low: float,
                  high: float, eps: float, delta: float, l0_sensitivity: float,
                  max_contributions_per_partition: float,
                  noise_kind: pipeline_dp.NoiseKind):
    """Helper function to compute the DP mean of a raw sum using the DP count.

    Args:
        count: Non-DP count.
        dp_count: DP count.
        sum: Non-DP sum.
        low, high: The lowest/highest contribution.
        eps, delta: The budget allocated.
        l0_sensitivity: The L0 sensitivity.
        max_contributions_per_partition: The maximum number of contributions
            per partition.
        noise_kind: The kind of noise used.

    Raises:
        ValueError: The noise kind is invalid.

    Returns:
        The anonymized mean.
    """
    middle = compute_middle(low, high)
    linf_sensitivity = max_contributions_per_partition * abs(middle - low)

    normalized_sum = sum - count * middle
    dp_normalized_sum = _add_random_noise(normalized_sum, eps, delta,
                                          l0_sensitivity, linf_sensitivity,
                                          noise_kind)
    return dp_normalized_sum / dp_count + middle


def compute_dp_mean(count: int, sum: float, dp_params: MeanVarParams):
    """Computes DP mean.

    Args:
        count: Non-DP count.
        sum: Non-DP sum.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.

    Returns:
        The tuple of anonymized count, sum and mean.
    """
    # Splits the budget equally between the two mechanisms.
    (count_eps, count_delta), (sum_eps, sum_delta) = equally_split_budget(
        dp_params.eps, dp_params.delta, 2)
    l0_sensitivity = dp_params.l0_sensitivity()

    dp_count = _add_random_noise(count, count_eps, count_delta, l0_sensitivity,
                                 dp_params.max_contributions_per_partition,
                                 dp_params.noise_kind)

    dp_mean = _compute_mean(count, dp_count, sum, dp_params.low, dp_params.high,
                            sum_eps, sum_delta, l0_sensitivity,
                            dp_params.max_contributions_per_partition,
                            dp_params.noise_kind)
    return dp_count, dp_mean * dp_count, dp_mean


def compute_dp_var(count: int, sum: float, sum_squares: float,
                   dp_params: MeanVarParams):
    """Computes DP variance.

    Args:
        count: Non-DP count.
        sum: Non-DP sum.
        sum_squares: Non-DP sum of squares.
        dp_params: The parameters used at computing the noise.

    Raises:
        ValueError: The noise kind is invalid.

    Returns:
        The tuple of anonymized count, sum, sum_squares and variance.
    """
    # Splits the budget equally between the three mechanisms.
    (count_eps,
     count_delta), (sum_eps,
                    sum_delta), (sum_squares_eps,
                                 sum_squares_delta) = equally_split_budget(
                                     dp_params.eps, dp_params.delta, 3)
    l0_sensitivity = dp_params.l0_sensitivity()

    dp_count = _add_random_noise(count, count_eps, count_delta, l0_sensitivity,
                                 dp_params.max_contributions_per_partition,
                                 dp_params.noise_kind)

    # Computes and adds noise to the mean.
    dp_mean = _compute_mean(count, dp_count, sum, dp_params.low, dp_params.high,
                            sum_eps, sum_delta, l0_sensitivity,
                            dp_params.max_contributions_per_partition,
                            dp_params.noise_kind)

    squares_low, squares_high = dp_params.squares_interval()

    # Computes and adds noise to the mean of squares.
    dp_mean_squares = _compute_mean(count, dp_count, sum_squares, squares_low,
                                    squares_high, sum_squares_eps,
                                    sum_squares_delta, l0_sensitivity,
                                    dp_params.max_contributions_per_partition,
                                    dp_params.noise_kind)

    dp_var = dp_mean_squares - dp_mean**2
    return dp_count, dp_mean * dp_count, dp_mean_squares * dp_count, dp_var
