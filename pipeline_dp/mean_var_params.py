"""DP computing of count, sum, mean, variance."""

import numpy as np

from dataclasses import dataclass
from pipeline_dp.aggregate_params import NoiseKind


@dataclass
class MeanVarParams:
	"""Manages the parameters used for computing the mean/variance."""
	eps: float
	delta: float
	low: float
	high: float
	sensitivity: float  # L1 for Laplace, L2 for Gaussian
	noise_kind: NoiseKind  # Laplace or Gaussian


def _apply_laplace_mechanism(value, eps, sensitivity):
	"""Applies the Laplace mechanism.

	Args:
		value: The result of querying the database.
		eps: Parameter of (epsilon, delta)-differential privacy.
		sensitivity: The L1-sensitivity of the query.
	"""
	return value + np.random.laplace(0, sensitivity / eps)


def _apply_gaussian_mechanism(value, eps, delta, sensitivity):
	"""Applies the Gaussian mechanism.

	Args:
		value: The result of querying the database.
		eps, delta: Parameters of (epsilon, delta)-differential privacy.
		sensitivity: The L2-sensitivity of the query.
	"""
	return value + np.random.normal(0, np.var(eps, delta, sensitivity))


def _add_random_noise(value, dp_params):
	"""Adds random noise to the value according to the given parameters.

	Args:
		value: Random noise is added to this value.
		dp_params: The parameters used at computing the noise.

	Raises:
		ValueError: The noise kind is invalid.
	"""
	if dp_params.noise_kind == NoiseKind.LAPLACE:
		return _apply_laplace_mechanism(value, dp_params.eps, dp_params.sensitivity)
	if dp_params.noise_kind == NoiseKind.GAUSSIAN:
		return _apply_gaussian_mechanism(value, dp_params.eps, dp_params.delta, dp_params.sensitivity)
	raise ValueError("Noise kind must be either Laplace or Gaussian.")


def compute_dp_count(count: int, dp_params: MeanVarParams):
	"""Computes DP count.

	Args:
		count: Non-DP count.
		dp_params: The parameters used at computing the noise.

	Raises:
		ValueError: The noise kind is invalid.
	"""
	return _add_random_noise(count, dp_params)


def compute_dp_sum(sum: float, dp_params: MeanVarParams):
	"""Computes DP sum.

	Args:
		sum: Non-DP sum.
		dp_params: The parameters used at computing the noise.

	Raises:
		ValueError: The noise kind is invalid.
	"""
	return _add_random_noise(sum, dp_params)


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
