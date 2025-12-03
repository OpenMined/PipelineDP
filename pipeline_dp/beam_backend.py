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
"""Apache Beam adapter."""

import itertools
import functools
from typing import Callable

try:
    import apache_beam as beam
    import apache_beam.transforms.combiners as combiners
except (AttributeError, ImportError):
    # It is fine if Apache Beam is not installed, other backends can be used.
    pass

import pipeline_dp.combiners as dp_combiners
from .pipeline_backend import PipelineBackend, _annotators


class UniqueLabelsGenerator:
    """Generates unique labels for each pipeline aggregation."""

    def __init__(self, suffix):
        self._labels = set()
        self._suffix = ("_" + suffix) if suffix else ""

    def _add_if_unique(self, label):
        if label in self._labels:
            return False
        self._labels.add(label)
        return True

    def unique(self, label):
        if not label:
            label = "UNDEFINED_STAGE_NAME"
        suffix_label = label + self._suffix
        if self._add_if_unique(suffix_label):
            return suffix_label
        for i in itertools.count(1):
            label_candidate = f"{label}_{i}{self._suffix}"
            if self._add_if_unique(label_candidate):
                return label_candidate


class BeamBackend(PipelineBackend):
    """Apache Beam adapter."""

    def __init__(self, suffix: str = ""):
        super().__init__()
        self._ulg = UniqueLabelsGenerator(suffix)

    @property
    def unique_lable_generator(self) -> UniqueLabelsGenerator:
        return self._ulg

    def to_collection(self, collection_or_iterable, col, stage_name: str):
        if isinstance(collection_or_iterable, beam.PCollection):
            return collection_or_iterable
        return col.pipeline | self._ulg.unique(stage_name) >> beam.Create(
            collection_or_iterable)

    def map(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Map(fn)

    def map_with_side_inputs(self,
                             col,
                             fn,
                             side_input_cols,
                             stage_name: str = None):
        side_inputs = [
            beam.pvalue.AsSingleton(side_input_col)
            for side_input_col in side_input_cols
        ]
        return col | self._ulg.unique(stage_name) >> beam.Map(fn, *side_inputs)

    def flat_map(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.FlatMap(fn)

    def flat_map_with_side_inputs(self, col, fn, side_input_cols,
                                  stage_name: str):
        side_inputs = [
            beam.pvalue.AsSingleton(side_input_col)
            for side_input_col in side_input_cols
        ]
        return col | self._ulg.unique(stage_name) >> beam.FlatMap(
            fn, *side_inputs)

    def map_tuple(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Map(lambda x: fn(*x))

    def map_values(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.MapTuple(lambda k, v:
                                                                   (k, fn(v)))

    def group_by_key(self, col, stage_name: str):
        """Groups the values for each key in the PCollection into a single sequence.
        Args:
          col: input collection with elements (key, value)
          stage_name: name of the stage
        Returns:
          A PCollection of tuples in which the type of the second item is an
          iterable, i.e. (key, Iterable[value]).
        """
        return col | self._ulg.unique(stage_name) >> beam.GroupByKey()

    def filter(self, col, fn, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Filter(fn)

    def filter_with_side_inputs(self, col, fn, side_input_cols,
                                stage_name: str):
        side_inputs = [
            beam.pvalue.AsSingleton(side_input_col)
            for side_input_col in side_input_cols
        ]
        return col | self._ulg.unique(stage_name) >> beam.Filter(
            fn, *side_inputs)

    def filter_by_key(self, col, keys_to_keep, stage_name: str):

        class PartitionsFilterJoin(beam.DoFn):

            def process(self, joined_data):
                key, rest = joined_data
                values, to_keep = rest.get(VALUES), rest.get(TO_KEEP)

                if not values:
                    return

                if to_keep:
                    for value in values:
                        yield key, value

        def does_keep(pk_val):
            return pk_val[0] in keys_to_keep

        # define constants for using as keys in CoGroupByKey
        VALUES, TO_KEEP = 0, 1

        if keys_to_keep is None:
            raise TypeError("Must provide a valid keys to keep")

        if isinstance(keys_to_keep, (list, set)):
            # Keys to keep are in memory.
            if not isinstance(keys_to_keep, set):
                keys_to_keep = set(keys_to_keep)
            return col | self._ulg.unique("Filtering out") >> beam.Filter(
                does_keep)

        # `keys_to_keep` are not in memory. Filter out with a join.
        keys_to_keep = (keys_to_keep | self._ulg.unique("Reformat PCollection")
                        >> beam.Map(lambda x: (x, True)))
        return ({
            VALUES: col,
            TO_KEEP: keys_to_keep
        } | self._ulg.unique("CoGroup by values and to_keep partition flag") >>
                beam.CoGroupByKey() | self._ulg.unique("Partitions Filter Join")
                >> beam.ParDo(PartitionsFilterJoin()))

    def keys(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Keys()

    def values(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Values()

    def sample_fixed_per_key(self, col, n: int, stage_name: str):
        return col | self._ulg.unique(
            stage_name) >> combiners.Sample.FixedSizePerKey(n)

    def count_per_element(self, col, stage_name: str):
        return col | self._ulg.unique(
            stage_name) >> combiners.Count.PerElement()

    def sum_per_key(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.CombinePerKey(sum)

    def combine_accumulators_per_key(self, col, combiner: dp_combiners.Combiner,
                                     stage_name: str):

        def merge_accumulators(accumulators):
            res = None
            for acc in accumulators:
                if res:
                    res = combiner.merge_accumulators(res, acc)
                else:
                    res = acc
            return res

        return col | self._ulg.unique(stage_name) >> beam.CombinePerKey(
            merge_accumulators)

    def reduce_per_key(self, col, fn: Callable, stage_name: str):
        combine_fn = lambda elements: functools.reduce(fn, elements)
        return col | self._ulg.unique(stage_name) >> beam.CombinePerKey(
            combine_fn)

    def flatten(self, cols, stage_name: str):
        return cols | self._ulg.unique(stage_name) >> beam.Flatten()

    def distinct(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Distinct()

    def reshuffle(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.Reshuffle()

    def to_list(self, col, stage_name: str):
        return col | self._ulg.unique(stage_name) >> beam.combiners.ToList()

    def annotate(self, col, stage_name: str, **kwargs):
        if not _annotators:
            return col
        for annotator in _annotators:
            col = annotator.annotate(col, self, self._ulg.unique(stage_name),
                                     **kwargs)
        return col
