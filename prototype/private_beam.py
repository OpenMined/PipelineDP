from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from typing import List, Optional, Union, Callable, Iterable

import apache_beam as beam
import apache_beam.transforms.combiners as combiners
from budget_accounting import BudgetAccountant, Budget
from data_structures import AggregateParams, Metrics, MetricsResult
from dataclasses import dataclass
from dp_utils import ThresholdCalculator, add_laplace_noise_for_vector
import numpy as np

# Abbreviations:
# pid - privacy id
# pk - partition key
# v - value
#
# After most pipeline operations elements of the output collection are specified
# in the comments. It is very convenient for understanding what is going on.


class ReportGenerator:
  """Collects information about a dp aggregation and generates report."""

  def __init__(self, name):
    self.name = name
    self.stages = []

  def add_stage(self, text):
    # text is str of function which returns str. The function is needed since in
    # case of lazy budget split, (eps, delta) per aggregation are not available
    # during the pipeline construction time.
    self.stages.append(text)

  def report(self) -> str:
    result = [f"Differential private {self.name}"]
    for i, stage_str in enumerate(self.stages):
      if hasattr(stage_str, "__call__"):  # this callable, i.e. lazy generation
        result.append(f"{i+1}. {stage_str()}")
      else:
        result.append(f"{i+1}. {stage_str}")
    return "\n".join(result)


@dataclass
class DataExtractors:
  privacy_id_extractor: Callable = None
  partition_extractor: Callable = None
  value_extractor: Callable = None


class PartitionsFilter(beam.DoFn):
  """Filters out unkonwn partitions for public partitions."""

  def process(self, join_result):
    # join result has format: (key, (values:Iterable, is_known:{None, 1})
    key = join_result[0]
    values = join_result[1][0]
    is_known = join_result[1][1]
    if values and is_known:
      for value in values:
        yield key, value


def sum_vectors(vectors):
  # assumed 'vectors' is iterable of the same length tuples/lists.
  size = len(vectors[0])
  res = [0] * size
  for v in vectors:
    for i in range(size):
      res[i] += v[i]
  return tuple(res)


def bound_by_l1_norm(vector, max_l1):
  l1 = sum(vector)
  if l1 <= max_l1:
    return vector
  coef = max_l1 / l1
  return tuple(t * coef for t in vector)


class PipelineOperations(ABC):

  @abstractmethod
  def is_lazy_execution(self):
    pass

  @abstractmethod
  def filter_partitions(self, col, agg_keys):  # todo: maybe better name
    pass

  @abstractmethod
  def map(self, col, fn, stage_name: str):
    pass

  @abstractmethod
  def flat_map(self, col, fn, stage_name: str):
    pass

  @abstractmethod
  def map_tuple(self, col, fn, stage_name: str):
    pass

  @abstractmethod
  def map_values(self, col, fn, stage_name: str):
    pass

  @abstractmethod
  def par_do(self, col, fn, stage_name: str):
    pass

  @abstractmethod
  def group_by_key_(self, col, stage_name: str):
    pass

  @abstractmethod
  def filter(self, col, fn, stage_name: str):
    pass

  @abstractmethod
  def keys(self, col, stage_name: str):
    pass

  @abstractmethod
  def values(self, col, stage_name: str):
    pass

  @abstractmethod
  def sample_fixed_per_key(self, col, n: int, stage_name: str):
    pass

  @abstractmethod
  def count_per_element(self, col, stage_name: str):
    pass

  @abstractmethod
  def add_zeros_values(self, col, public_partitions, zero_fn):
    pass


class BeamBackend(PipelineOperations):

  def is_lazy_execution(self):
    return True

  def map(self, col, fn, stage_name: str):
    return col | stage_name >> beam.Map(fn)

  def flat_map(self, col, fn, stage_name: str):
    return col | stage_name >> beam.FlatMap(fn)

  def map_tuple(self, col, fn, stage_name: str):
    return col | stage_name >> beam.MapTuple(fn)

  def map_values(self, col, fn, stage_name: str):
    return col | stage_name >> beam.MapTuple(lambda k, v: (k, fn(v)))

  def par_do(self, col, fn, stage_name: str):
    return col | stage_name >> beam.ParDo(fn)

  def group_by_key_(self, col, stage_name: str):
    return col | stage_name >> beam.GroupByKey()

  def filter(self, col, fn, stage_name: str):
    return col | stage_name >> beam.Filter(fn)

  def keys(self, col, stage_name: str):
    return col | stage_name >> beam.Keys()

  def values(self, col, stage_name: str):
    return col | stage_name >> beam.Values()

  def sample_fixed_per_key(self, col, n: int, stage_name: str):
    return col | stage_name >> combiners.Sample.FixedSizePerKey(n)

  def count_per_element(self, col, stage_name: str):
    return col | stage_name >> combiners.Count.PerElement()

  def filter_partitions(self, col, agg_keys):
    # Assumes col format (key, value)
    # CoGroupByKey requires (key, value)s, that's why 1 is used
    agg_keys = agg_keys | beam.Map(lambda x: (x, 1))

    return {
        0: col,
        1: agg_keys
    } | beam.CoGroupByKey() | beam.ParDo(PartitionsFilter())

  def add_zeros_values(self, pcoll, public_partitions, zero_fn):
    if not isinstance(public_partitions, beam.pvalue.PCollection):
      public_partitions = pcoll.pipeline | "To Pcollection" >> beam.Create(
          public_partitions)

    zero_pcoll = public_partitions | "Zero elements for public_partitions" >> beam.Map(
        lambda partition: (partition, zero_fn()))

    return (pcoll, zero_pcoll) | "Add zero elements" >> beam.Flatten()


class LocalBackend(PipelineOperations):

  def is_lazy_execution(self):
    return False

  def map(self, col, fn, stage_name: str):
    return list(map(fn, col))

  def flat_map(self, col, fn, stage_name: str):
    return list([x for el in col for x in fn(el)])

  def map_tuple(self, col, fn, stage_name: str):
    return [fn(k, v) for k, v in col]

  def map_values(self, col, fn, stage_name: str):
    return [(k, fn(v)) for k, v in col]

  def par_do(self, col, fn, stage_name: str):
    res = []
    for el in col:
      res.extend(list(fn(el)))
    return res

  def group_by_key_(self, col, stage_name: str):
    d = defaultdict(lambda: [])
    for key, value in col:
      d[key].append(value)

    return list(d.items())

  def filter(self, col, fn, stage_name: str):
    return list(filter(fn, col))

  def keys(self, col, stage_name: str):
    return list(map(lambda kv: kv[0], col))

  def values(self, col, stage_name: str):
    return list(map(lambda kv: kv[1], col))

  def sample_fixed_per_key(self, col, n: int, stage_name: str):
    d = defaultdict(lambda: [])
    for k, v in col:
      d[k].append(v)

    result = []
    for k, values in d.items():
      if len(values) <= n:
        result.append((k, values))
        continue
      # random.choice doesn't work with list of tuples, so it's needed to make
      # choice over indices.
      sampled_indices = np.random.choice(range(len(values)), n, replace=False)
      sampled_values = [values[i] for i in sampled_indices]
      # sampled_values = list(np.random.choice(values, n, replace=False))
      result.append((k, sampled_values))
    return result

  def count_per_element(self, col, stage_name: str):
    d = defaultdict(lambda: 0)
    for el in col:
      d[el] += 1
    return list(d.items())

  def filter_partitions(self, col, agg_keys):
    # Assumes pcoll format (key, value)
    agg_keys = set(agg_keys)
    return [el for el in col if el[0] in agg_keys]

  def add_zeros_values(self, col, public_partitions, zero_fn):
    result = [el for el in col]
    for partition in public_partitions:
      result.append((partition, zero_fn()))
    return result


class DPEngine:

  def __init__(self, budget_accountant: BudgetAccountant,
               ops: PipelineOperations):
    self._budget_accountant = budget_accountant
    self._ops = ops
    self._report_generators = []

  def aggregate(self, col, params: AggregateParams,
                data_extractors: DataExtractors):
    self._report_generators.append(
        ReportGenerator(
            f"computing metrics: {[m.value[0] for m in params.metrics]}"))
    partition_budget, noise_budget = self._split_budget(
        params.budget_weight, params.public_partitions is None, False)

    col = self._extract_data(col, data_extractors)
    # (pid, pk, v)

    if params.preagg_partition_selection:
      col = self._ops.map(
          col, lambda pid_pk_v: (pid_pk_v[1], (pid_pk_v[0], pid_pk_v[2])),
          "To (pk, (pid, v))")
      # (pk, (pid, v))

      col = self._select_preagg_partitions(col, params.public_partitions,
                                           partition_budget)
      # (pk, (pid, v))

      col = self._ops.map_tuple(col, lambda pk, pid_v:
                                ((pid_v[0], pk), pid_v[1]), "To ((pid, pk), v)")
      # ((pid, pk), v)
    else:
      col = self._ops.map(
          col, lambda pid_pk_v: ((pid_pk_v[0], pid_pk_v[1]), pid_pk_v[2]),
          "To ((pid, pk), v)")
      # ((pid, pk), v)

    # Clipping
    self._add_report_stage(f"Clip values to {params.low, params.high}")
    col = self._ops.map_values(col,
                               lambda v: np.clip(v, params.low, params.high),
                               "Clipping values")
    # ((pid, pk), v)

    # Convert to vector
    col = self._value_to_vector(col, params.metrics)
    # ((pid, pk), vector)

    col = self._limit_contribution_and_sum_vectors_per_key(
        col, params.max_partitions_contributed)
    # ((pid, pk), vector)

    col = self._ops.map_tuple(col, lambda pid_pk, v: (pid_pk[0],
                                                      (pid_pk[1], v)),
                              "To (pid, (pk, vector))")
    # (pid, (pk, vector))

    col = self._bound_cross_partition_contributions(
        col, params.max_partitions_contributed)
    # (pk, vector)

    if params.public_partitions is not None:
      self._add_report_stage(f"Adding zero results to missing public partition")
      col = self._ops.add_zeros_values(
          col, params.public_partitions, zero_fn=lambda: (0,) * 4
      )  # there are 4 metrics: privacy_id_count, count, sum, sum of squares

    # (pk, vector)
    col = self._sum_vectors_per_key(col)

    if not params.preagg_partition_selection:
      col = self._select_postagg_partitions(col, params.public_partitions,
                                            partition_budget)
    # (pk, vector)

    col = self._add_noise_to_vector(col, noise_budget, params)
    # (pk, vector)

    col = self._vector_to_metrics(col, params.metrics)
    # (pk, MetricsResult)

    return col

  def _add_report_stage(self, text):
    self._report_generators[-1].add_stage(text)

  def _extract_data(self, col, data_extractors):
    """Converts to col of (pid, pk, v)"""

    def input_data_converter(x):
      pid = data_extractors.privacy_id_extractor(x)
      pk = data_extractors.partition_extractor(x)
      v = data_extractors.value_extractor(x)
      return pid, pk, v

    return self._ops.map(col, input_data_converter, "Extract data")

  def _extract_pid_value(self, col, data_extractors):
    """Returns collection (pid, v)"""
    pid_extractor = data_extractors.privacy_id_extractor
    value_extractor = data_extractors.value_extractor
    return self._ops.map(col, lambda x: (pid_extractor(x), value_extractor(x)),
                         "extract data")

  def _split_budget(self, budget_weight, non_public_partitions: bool,
                    delta_for_noise: bool):
    """Returns public and noise budget."""
    if non_public_partitions:
      if delta_for_noise:
        return self._get_budget("Partition selection", budget_weight / 2,
                                budget_weight / 2), self._get_budget(
                                    "Noise", budget_weight / 2,
                                    budget_weight / 2)
      return self._get_budget("Partition selection", budget_weight / 2,
                              budget_weight), self._get_budget(
                                  "Noise", budget_weight / 2, 0)

    noise_delta = budget_weight if delta_for_noise else 0
    return None, self._get_budget("Noise", budget_weight, noise_delta)

  def _select_preagg_partitions(self, col, public_partitions, budget: Budget):
    # col: (pk, (pid, v))
    report_str = "Partitions selection: "
    if public_partitions is None:
      report_str += "using thresholding"
      partitions = self._select_private_partition_keys(col, budget)
    else:
      report_str += " using provided public partition"
      partitions = public_partitions

    self._add_report_stage(report_str)

    # pk -> (pid, v)
    return self._ops.filter_partitions(col, partitions)

  def _select_postagg_partitions(self, col, public_partitions, budget: Budget):
    # col: (pk, vector), vector[0] - number of unique counts
    report_str = "Post aggregation partitions selection: "
    if public_partitions is None:
      report_str += "using thresholding"
      pk_count = self._ops.map_tuple(col, lambda pk, v: (pk, v[0]),
                                     "To (pk, privacy_id_count)")
      partitions = self._get_private_partition_keys_from_privacy_id_counts(
          pk_count, budget)
    else:
      report_str += " using provided public partition"
      partitions = public_partitions

    self._add_report_stage(report_str)

    # pk -> (pid, v)
    return self._ops.filter_partitions(col, partitions)

  def _select_private_partition_keys(self, input, budget: Budget):
    """Performs privacy partition keys procedure and returns selected partition

     keys.
    """
    # input: (pk, (pid, v))

    col = self._ops.map_tuple(input, lambda pk, pid_v: (pid_v[0], pk),
                              "To (pid, pk)")
    # (pid, pk)

    col = self._ops.sample_fixed_per_key(col, 1,
                                         "Take 1 contribution per privacy_id")
    # (pid:[pk])

    col = self._ops.flat_map(col, lambda kv: kv[1], "Leave only pk")
    # pk

    col = self._ops.count_per_element(col, "Count pk")
    # (pk, count)

    return self._get_private_partition_keys_from_privacy_id_counts(col, budget)

  def _get_private_partition_keys_from_privacy_id_counts(self, col, budget: Budget):
    # input: (pk, privacy_id_count)

    calculator = ThresholdCalculator(budget)

    def filter_fn(ak_count):
      return calculator.does_keep(ak_count[1])

    col = self._ops.filter(col, filter_fn, "Select non-public pk")
    # (pk, count)

    return self._ops.keys(col, "")  # pk

  def _limit_contribution_and_sum_vectors_per_key(self, col, n: int):
    self._add_report_stage(f"Per-partition contribution: randomly selected not "
                           f"more than {n} contributions")

    col = self._ops.sample_fixed_per_key(col, n, "Sample per (pid, pk)")
    # (key, [vector])
    col = self._ops.map_values(col, sum_vectors, "Sum per (pid, pk)")
    return self._ops.map_values(col, lambda v: (1,) + v, "Add unique pid count")

  def _bound_cross_partition_contributions(self, col, max_partitions: int):
    # Input:(pid, (pk, v))
    self._add_report_stage(
        f"Contribution bonding: randomly selected not more than "
        f"{max_partitions} partitions per user")

    col = self._ops.sample_fixed_per_key(col, max_partitions, "Sample per pk")
    # (pid: [(pk, v)])

    return self._ops.flat_map(col, lambda kv: kv[1], "Unnest values")

  def _value_to_vector(self, col, metrics):
    return self._ops.map_values(col, lambda v: (1, v, v * v), "To vector")

  def _sum_vectors_per_key(self, col):
    col = self._ops.group_by_key_(col, "Group by pk")
    col = self._ops.map_values(col, sum_vectors, "Sum per pk")
    return col

  @dataclass
  class AddNoiseParams:
    max_partitions_contributed: int
    max_contributions_per_partition: int
    low: float
    high: float

  def _add_noise_to_vector(self, col, budget: Budget, params: AddNoiseParams):

    def get_noise_beta(eps):
      # TODO: split the budget to a subset of metrics is computed, eg. if only
      # count and sum are to be computed, no need to anonymize unique count
      n_metrics = 4
      return [
          sens_privacy_id_count / eps * n_metrics,
          sens_count / eps * n_metrics,
          sens_sum / eps * n_metrics,
          sens_sum2 / eps * n_metrics,
      ]

    def report_str_fn():
      # str generation must be lazy, because eps is available only after pipeline is contructed.
      noise_beta = get_noise_beta(budget.eps)
      noise_std = [b * np.sqrt(2) for b in noise_beta]
      return f"Adding laplace random noise with scale={noise_beta} " \
             f"(std={noise_std}) to (privacy_id_count, count, sum, " \
             f"sum_squares) per partition."

    self._add_report_stage(report_str_fn)

    max_value = max(abs(params.low), abs(params.high))
    sens_privacy_id_count = params.max_partitions_contributed
    sens_count = params.max_contributions_per_partition * params.max_partitions_contributed
    sens_sum = max_value * sens_count
    sens_sum2 = max_value**2 * sens_count

    def add_noise_fn(x):
      noise_beta = get_noise_beta(budget.eps)
      return add_laplace_noise_for_vector(x, noise_beta)

    return self._ops.map_values(col, add_noise_fn, "Add laplace noise")

  def _vector_to_metrics(self, col, metrics: Iterable[Metrics]):
    out_privacy_id_count = Metrics.PRIVACY_ID_COUNT in metrics
    out_count = Metrics.COUNT in metrics
    out_sum = Metrics.SUM in metrics
    out_mean = Metrics.MEAN in metrics
    out_var = Metrics.VAR in metrics

    def compute_metrics(v):
      privacy_id_count, count, sum, sum2 = v
      res = MetricsResult()
      if out_privacy_id_count:
        res.privacy_id_count = privacy_id_count
      if out_count:
        res.count = count
      if out_sum:
        res.sum = sum
      mean = sum / count if count else 0
      if out_mean:
        res.mean = mean
      if out_var:
        if count == 0:
          res.var = 0
        else:
          res.var = sum2 / count - mean**2
      return res

    return self._ops.map_values(col, compute_metrics, "Base metrics to metrics")

  def _get_budget(self, text, eps_weight, delta_weight):
    if self._ops.is_lazy_execution():
      return self._budget_accountant.request_budget(eps_weight, delta_weight,
                                                    text)
    return self._budget_accountant.use_budget(eps_weight, delta_weight, text)
