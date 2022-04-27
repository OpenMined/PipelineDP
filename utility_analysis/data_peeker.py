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
"""Main class to sample data sketches for privacy utility analysis in DP.

This file contains helper classes to sample raw data for utility analysis in
Differential Privacy (DP). Note that most of the operations in this file are not
DP operations and the results contain the raw data. Users should mainly use the
functions to tune parameters and should not share any data generated if they
want DP guarantees.

The classes of this file can work with the corresponding Colab notebook to
generate visualizations. But they are also self-contained and can be used
directly.
"""

import dataclasses
import functools
from typing import Any, Sequence, Union, Optional, Tuple

import pipeline_dp

from utility_analysis import non_private_combiners

DataType = Union[Sequence[Any]]
try:
    from pyspark import RDD
    DataType = Union[DataType, RDD]
except ImportError:
    pass

try:
    from apache_beam import pvalue
    DataType = Union[DataType, pvalue.PCollection]
except ImportError:
    pass

@dataclasses.dataclass(frozen=True)
class SampleParams:
    number_of_sampled_partitions: int
    metrics: Optional[Sequence[pipeline_dp.Metrics]] = None


def _extract_fn(data_extractors: pipeline_dp.DataExtractors,
                row: DataType) -> DataType:
    """Extracts the columns to (pid, pkey, pvalue).

    Args:
      data_extractors: A function to extract privacy_id, partition_key, value of
        the input data.
      row: The data to extract, should usually be raw input of the pipline.

    Returns:
      Data in format of (pid, pkey, pvalue) defined by the extractors.
    """
    return data_extractors.privacy_id_extractor(
        row), data_extractors.partition_extractor(
            row), data_extractors.value_extractor(row)


class DataPeeker:
    """A helper class that contains methods to for privacy utility analysis."""

    def __init__(self, ops: pipeline_dp.pipeline_backend.PipelineBackend):
        self._be = ops

    def sketch(self, input_data: DataType, params: SampleParams,
               data_extractors: pipeline_dp.DataExtractors) -> DataType:
        """Generates sketches in the format of (partition_key, value, partition_count).

        The sketches has one entry for each unique (partition_key, privacy_id).
        Parameter tuning on outputs of sketch ignores `min_value` and `max_value` of
        AggregateParams

        partition_key: the hashed version of the current partition key
        partition_value: the per privacy id per partition_key aggregated value
        partition_count: the number of partitions this privacy id contributes to

        Args:
          input_data: The data to sample. It can be local data, beam PCollection or
            Spark RDD depending on the engine used.
          params: The parameters defining sampling properties.
          data_extractors: A function to extract privacy_id, partition_key, value of
            the input data.

        Returns:
          Sketches in the format of (partition_key, value, partition_count).
        """
        if params.metrics is None:
            raise ValueError("Must provide aggregation metrics for sketch.")
        if len(params.metrics) != 1 or params.metrics[0] not in [
                pipeline_dp.aggregate_params.Metrics.SUM,
                pipeline_dp.aggregate_params.Metrics.COUNT
        ]:
            raise ValueError(
                "Sketch only supports a single aggregation and it must be COUNT or SUM."
            )
        combiner = non_private_combiners.create_compound_combiner(
            metrics=params.metrics)

        # Extract the columns.
        col = self._be.map(input_data,
                           functools.partial(_extract_fn, data_extractors),
                           "Extract (privacy_id, partition_key, value))")
        # col : (privacy_id, partition_key, value)
        col = self._be.map_tuple(
            col, lambda pid, pk, v: (pk, (pid, v)),
            "Rekey to (partition_key, (privacy_id, value))")
        # col : (partition_key, (privacy_id, value))
        # sample
        # group by key, filter keys by sampling, expand the values by flat map
        col = self._be.group_by_key(col, "Group by pk")
        col = self._be.map_tuple(col, lambda pk, pid_v_seq: (1,
                                                             (pk, pid_v_seq)),
                                 "Rekey to (1, (pk, pid_v_seq))")
        col = self._be.sample_fixed_per_key(col,
                                            params.number_of_sampled_partitions,
                                            "Sample partitions")
        col = self._be.flat_map(col, lambda plst: plst[1], "Extract values")

        def flatten_sampled_results(
            pk_pid_pval_list: Tuple[Any, Sequence[Tuple[Any, Any]]]
        ) -> Sequence[Tuple[Any, Tuple[Any, Any]]]:
            pk, pid_pval_list = pk_pid_pval_list
            return [(pk, pid_pval) for pid_pval in pid_pval_list]

        col = self._be.flat_map(col, flatten_sampled_results,
                                "Flatten to (pk, (pid, value))")

        # col : (partition_key, (privacy_id, value))
        # calculates partition_count after sampling and per
        # (partition_key, privacy_id) pair aggregated value
        col = self._be.map_tuple(col, lambda pk, pid_v: (
            (pk, pid_v[0]), pid_v[1]), "Transform to (pk, pid), value))")
        # col : ((partition_key, privacy_id), value)

        col = self._be.group_by_key(col, "Group by (pk, pid)")
        # col : ((partition_key, privacy_id), [value])
        col = self._be.map_values(col, combiner.create_accumulator,
                                  "Aggregate by (pk, pid)")
        # col : ((partition_key, privacy_id), accumulator)
        col = self._be.map_tuple(
            col, lambda pk_pid, p_value: (pk_pid[1], (pk_pid[0], p_value)),
            "Transform to (pid, (pk, accumulator))")
        # col : (privacy_id, (partition_key, accumulator))
        col = self._be.group_by_key(col, "Group by privacy_id")

        key_accumulator_sequence_type = Sequence[Tuple[
            Any, pipeline_dp.accumulator.Accumulator]]

        def calculate_partition_count(
            key_accumulator_list: key_accumulator_sequence_type
        ) -> Tuple[int, key_accumulator_sequence_type]:
            partition_count = len(set(pk for pk, _ in key_accumulator_list))
            return (partition_count, key_accumulator_list)

        col = self._be.map_values(col, calculate_partition_count,
                                  "Calculates partition_count")

        # col : (privacy_id, (partition_count, [(partition_key, accumulator)]))

        def flatten_results(
            input_col: Tuple[Any, Tuple[int, key_accumulator_sequence_type]]
        ) -> Sequence[Tuple[Any, Any, int]]:
            _, pcount_pk_acc_list = input_col
            pcount, pk_acc_list = pcount_pk_acc_list
            return [(pk, acc[0], pcount) for pk, acc in pk_acc_list]

        return self._be.flat_map(
            col, flatten_results,
            "Flatten to (pk, aggregated_value, partition_count)")
        # (partition_key, aggregated_value, partition_count)

    def sample(self, input_data: DataType, params: SampleParams,
               data_extractors: pipeline_dp.DataExtractors) -> DataType:
        """Generates sampled outputs of the input data according to sample parameters.

        The sampling is by partitions. e.g. a certain amount of partitions_keys are
        selected and the output contains all records with these partition_keys.

        Args:
          input_data: The data to sample. It can be local data, beam PCollection or
            Spark RDD depending on the engine used.
          params: The parameters defining sampling properties.
          data_extractors: A function to extract privacy_id, partition_key, value of
            the input data.

        Returns:
          Sampled output containing tuple of (privacy_id, partition_key, value).
        """

        col = self._be.map(input_data,
                           functools.partial(_extract_fn, data_extractors),
                           "Extract (privacy_id, partition_key, value))")
        # col : (privacy_id, partition_key, value)
        col = self._be.map_tuple(
            col, lambda pid, pk, v: (pk, (pid, v)),
            "Rekey to (partition_key, (privacy_id, value))")
        # col : (partition_key, (privacy_id, value))
        # Sample the data.
        # group by key, filter keys by sampling, expand the values by flat map
        col = self._be.group_by_key(col, "Group by pk")
        col = self._be.map_tuple(col, lambda pk, pid_v_seq: (1,
                                                             (pk, pid_v_seq)),
                                 "Rekey to (1, (pk, pid_v_seq))")
        col = self._be.sample_fixed_per_key(col,
                                            params.number_of_sampled_partitions,
                                            "Sample partitions")
        col = self._be.flat_map(col, lambda plst: plst[1], "")

        def expand_fn(pk_pidandvseq: DataType):
            pk, pid_pv_seq = pk_pidandvseq
            return [(pid, pk, v) for pid, v in pid_pv_seq]

        col = self._be.flat_map(col, expand_fn, "Transform to (pid, pk, value)")
        return col

    def aggregate_true(self, col, params: SampleParams,
                       data_extractors: pipeline_dp.DataExtractors) -> DataType:
        """Computes raw aggregation results of the input data without adding noises.

        Aggregation means aggregate values group by partition_key. Both values and
        partition_key are extracted by data extractors.

        Args:
          input_data: The data to sample. It can be local data, beam PCollection or
            Spark RDD depending on the engine used.
          data_extractors: A function to extract privacy_id, partition_key, value of
            the input data.

        Returns:
          True aggregation results.
        """
        combiner = non_private_combiners.create_compound_combiner(
            metrics=params.metrics)

        col = self._be.map(
            col, lambda row: (data_extractors.privacy_id_extractor(row),
                              data_extractors.partition_extractor(row),
                              data_extractors.value_extractor(row)),
            "Extract (privacy_id, partition_key, value))")
        # col : (privacy_id, partition_key, value)
        col = self._be.map_tuple(
            col, lambda pid, pk, v: ((pid, pk), v),
            "Rekey to ( (privacy_id, partition_key), value))")
        col = self._be.group_by_key(col, "Group by pk")
        col = self._be.map_values(col, combiner.create_accumulator,
                                  "Aggregate by (pk, pid)")
        # ((privacy_id, partition_key), aggregator)
        col = self._be.map_tuple(col, lambda pid_pk, v: (pid_pk[1], v),
                                 "Drop privacy id")
        # col : (partition_key, accumulator)
        col = self._be.combine_accumulators_per_key(
            col, combiner, "Reduce accumulators per partition key")
        # col : (partition_key, accumulator)
        # Compute metrics.
        col = self._be.map_values(col, combiner.compute_metrics,
                                  "Compute DP metrics")
        # col : (partition_key, aggregated_value)
        return col
