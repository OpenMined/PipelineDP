import numpy as np
from dataclasses import dataclass
from budget_accounting import Budget
from data_structures import Metrics
from typing import Iterable, List


def calculate_keep_partitions_probabilities(eps, delta, max_buckets=1):
  # this is a simple implementation of https://arxiv.org/pdf/2006.03684.pdf
  # https://github.com/google/differential-privacy/blob/main/cc/algorithms/partition-selection.h C++ implementation
  eps, delta = eps / max_buckets, delta / max_buckets  # todo: more effective when max_buckets > 1
  assert delta > 0, "Delta must be positive for thresholding with unknown partitions"
  a = [0]
  alpha = np.exp(eps)
  while True:
    x = a[-1] * alpha + delta
    if x >= 0.5:
      start = len(a) - 1
      if x < 1 - a[-1]:
        a.append(0.5)
      break
    a.append(x)

  for i in range(start, 0, -1):
    a.append(1 - a[i])

  return a


class ThresholdCalculator:
  def __init__(self, budget: Budget):
    self._budget = budget
    self._keep_probs = []

  def does_keep(self, n: int):
    if not self._keep_probs:
      self._keep_probs = calculate_keep_partitions_probabilities(self._budget.eps, self._budget.delta)
    if n <= 0: return False
    if n >= len(self._keep_probs): return True
    return np.random.binomial(1, self._keep_probs[n])

def add_laplace_noise_for_vector(vector: list, noise_b: list):
  noisifed_vector = [0] * len(vector)
  for i, v in enumerate(vector):
    # todo: secure noise should be used instead of np noise.
    noisifed_vector[i] = v + np.random.laplace(0, noise_b[i])
  return noisifed_vector
