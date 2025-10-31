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
import unittest
from typing import Iterable, List
import pipeline_dp
import pipeline_dp.combiners as dp_combiners
from pipeline_dp import DataExtractors
from pipeline_dp.pipeline_backend import LocalBackend, LazySingleton


class TestLazySingleton(unittest.TestCase):

    def test_with_list(self):
        ls = LazySingleton([10])
        # Also check immediate retrieval
        self.assertEqual(ls.singleton(), 10)

    def test_init_with_valid_iterable_generator(self):
        data = (x for x in [10])  # lazy generator
        ls = LazySingleton(data)
        self.assertIsNone(ls._singleton)  # Should be lazy
        self.assertIsInstance(ls._iterable, Iterable)
        self.assertEqual(ls.singleton(), 10)
        self.assertEqual(ls.singleton(), 10)

    def test_init_with_empty_list_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "exactly one element.*found 0"):
            LazySingleton([])

    def test_init_with_multi_element_list_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "exactly one element.*found 3"):
            LazySingleton([1, 2, 3])

    def test_init_with_non_iterable_raises_type_error(self):
        with self.assertRaisesRegex(TypeError,
                                    "must be a list or an Iterable.*got int"):
            LazySingleton(123)


class LocalBackendTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = LocalBackend()
        cls.data_extractors = DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])

    def test_to_multi_transformable_collection(self):
        col = range(5)
        self.assertEqual(list(col), [0, 1, 2, 3, 4])
        self.assertEqual(list(col), [0, 1, 2, 3, 4])

    def test_map(self):
        self.assertEqual(list(self.backend.map([], lambda x: x / 0)), [])

        self.assertEqual(list(self.backend.map([1, 2, 3], str)),
                         ["1", "2", "3"])
        self.assertEqual(list(self.backend.map(range(5), lambda x: x**2)),
                         [0, 1, 4, 9, 16])

    def test_map_with_side_inputs(self):
        col = [1, 2]
        # side input must be 1-element iterable.
        side_input1 = [5]
        side_input2 = (x for x in [30])  # lazy iterable
        add_fn = lambda x, s1, s2: x + s1 + s2

        result = self.backend.map_with_side_inputs(col, add_fn,
                                                   [side_input1, side_input2],
                                                   "map_with_side_inputs")

        expected_result = [36, 37]
        self.assertEqual(list(result), expected_result)

    def test_map_with_side_inputs_raises_not_singleton(self):
        col = [1, 2]
        side_input = [10, 20]  # it should be 1 element

        add_fn = lambda x, s: x + s
        with self.assertRaises(ValueError):
            result = self.backend.map_with_side_inputs(col, add_fn,
                                                       [side_input],
                                                       "map_with_side_inputs")

    def test_flat_map_with_side_inputs(self):
        col = [[1, 2], [3]]
        # side input must be 1-element iterable.
        side_input1 = [5]
        side_input2 = (x for x in [30])  # lazy iterable

        def add_fn(elems: list, s1, s2):
            for elem in elems:
                yield elem + s1 + s2

        result = self.backend.flat_map_with_side_inputs(
            col, add_fn, [side_input1, side_input2],
            "flat_map_with_side_inputs")

        expected_result = [36, 37, 38]
        self.assertEqual(list(result), expected_result)

    def test_map_tuple(self):
        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(
            list(self.backend.map_tuple(tuple_list, lambda k, v: k + v)),
            [3, 5, 7])

        self.assertEqual(
            list(
                self.backend.map_tuple(tuple_list, lambda k, v:
                                       (str(k), str(v)))), [("1", "2"),
                                                            ("2", "3"),
                                                            ("3", "4")])

    def test_map_values(self):
        self.assertEqual(list(self.backend.map_values([], lambda x: x / 0)), [])

        tuple_list = [(1, 2), (2, 3), (3, 4)]

        self.assertEqual(list(self.backend.map_values(tuple_list, str)),
                         [(1, "2"), (2, "3"), (3, "4")])
        self.assertEqual(
            list(self.backend.map_values(tuple_list, lambda x: x**2)),
            [(1, 4), (2, 9), (3, 16)])

    def test_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.backend.group_by_key(some_dict)),
                         [("cheese", ["brie", "swiss"]),
                          ("bread", ["sourdough"])])

    def test_filter(self):
        self.assertEqual(list(self.backend.filter([], lambda x: True)), [])
        self.assertEqual(list(self.backend.filter([], lambda x: False)), [])

        example_list = [1, 2, 2, 3, 3, 4, 2]

        self.assertEqual(
            list(self.backend.filter(example_list, lambda x: x % 2)), [1, 3, 3])
        self.assertEqual(
            list(self.backend.filter(example_list, lambda x: x < 3)),
            [1, 2, 2, 2])

    def test_filter_with_side_inputs(self):
        data = [1, 2, 3, 4, 5, 6]
        side_input = [[2, 4]]  # side_input must be singleton (i.e. 1 element).

        def filter_fn(x, side_input):
            return x in side_input

        result = list(
            self.backend.filter_with_side_inputs(data, filter_fn, [side_input]))
        self.assertEqual(result, [2, 4])

    def test_filter_by_key_empty_keys_to_keep(self):
        col = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10)]
        keys_to_keep = []
        result = self.backend.filter_by_key(col, keys_to_keep, "filter_by_key")
        self.assertEqual(list(result), [])

    def test_filter_by_key_remove(self):
        col = [(7, 1), (2, 1), (3, 9), (4, 1), (9, 10)]
        keys_to_keep = [7, 9]
        result = self.backend.filter_by_key(col, keys_to_keep, "filter_by_key")
        self.assertEqual(list(result), [(7, 1), (9, 10)])

    def test_keys(self):
        self.assertEqual(list(self.backend.keys([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.backend.keys(example_list)), [1, 2, 3, 4])

    def test_values(self):
        self.assertEqual(list(self.backend.values([])), [])

        example_list = [(1, 2), (2, 3), (3, 4), (4, 8)]

        self.assertEqual(list(self.backend.values(example_list)), [2, 3, 4, 8])

    def test_count_per_element(self):
        example_list = [1, 2, 3, 4, 5, 6, 1, 4, 0, 1]
        result = self.backend.count_per_element(example_list)

        self.assertEqual(dict(result), {
            1: 3,
            2: 1,
            3: 1,
            4: 2,
            5: 1,
            6: 1,
            0: 1
        })

    def test_sum_per_key(self):
        data = [(1, 2), (2, 1), (1, 4), (3, 8), (2, -3), (10, 5)]
        result = list(self.backend.sum_per_key(data))
        self.assertEqual(result, [(1, 6), (2, -2), (3, 8), (10, 5)])

    def test_combine_accumulators_per_key(self):
        data = [(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)]
        col = self.backend.group_by_key(data)
        sum_combiner = SumCombiner()
        col = self.backend.map_values(col, sum_combiner.create_accumulator)
        col = self.backend.combine_accumulators_per_key(col, sum_combiner)
        col = self.backend.map_values(col, sum_combiner.compute_metrics)
        result = list(col)
        self.assertEqual(result, [(1, 6), (2, 4), (3, 8)])

    def test_reduce_per_key(self):
        data = [(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)]
        col = self.backend.reduce_per_key(data, lambda x, y: x + y, "Reduce")
        result = list(col)
        self.assertEqual(result, [(1, 6), (2, 4), (3, 8)])

    def test_to_list(self):
        data = [1, 2, 3, 4, 5]
        col = self.backend.to_list(data, "To list")
        result = list(col)
        self.assertEqual(result, [[1, 2, 3, 4, 5]])

    def test_laziness(self):

        def exceptions_generator_function():
            yield 1 / 0

        def assert_laziness(operator, *args):
            try:
                operator(exceptions_generator_function(), *args)
            except ZeroDivisionError:
                self.fail(f"local {operator.__name__} is not lazy")

        # reading from exceptions_generator_function() results in error:
        self.assertRaises(ZeroDivisionError, next,
                          exceptions_generator_function())

        # lazy operators accept exceptions_generator_function()
        # as argument without raising errors:
        assert_laziness(self.backend.map, str)
        assert_laziness(self.backend.map_values, str)
        assert_laziness(self.backend.filter, bool)
        assert_laziness(self.backend.values)
        assert_laziness(self.backend.keys)
        assert_laziness(self.backend.count_per_element)
        assert_laziness(self.backend.sum_per_key)
        assert_laziness(self.backend.flat_map, str)
        assert_laziness(self.backend.sample_fixed_per_key, int)
        assert_laziness(self.backend.filter_by_key, [1, 2])
        assert_laziness(self.backend.distinct, str)

    def test_sample_fixed_per_key_requires_no_discarding(self):
        input_col = [("pid1", ('pk1', 1)), ("pid1", ('pk2', 1)),
                     ("pid1", ('pk3', 1)), ("pid2", ('pk4', 1))]
        n = 3

        sample_fixed_per_key_result = list(
            self.backend.sample_fixed_per_key(input_col, n))

        expected_result = [("pid1", [('pk1', 1), ('pk2', 1), ('pk3', 1)]),
                           ("pid2", [('pk4', 1)])]
        self.assertEqual(sample_fixed_per_key_result, expected_result)

    def test_sample_fixed_per_key_with_sampling(self):
        input_col = [(("pid1", "pk1"), 1), (("pid1", "pk1"), 1),
                     (("pid1", "pk1"), 1), (("pid1", "pk1"), 1),
                     (("pid1", "pk1"), 1), (("pid1", "pk2"), 1),
                     (("pid1", "pk2"), 1)]
        n = 3

        sample_fixed_per_key_result = list(
            self.backend.sample_fixed_per_key(input_col, n))

        self.assertTrue(
            all(
                map(lambda pid_pk_v: len(pid_pk_v[1]) <= n,
                    sample_fixed_per_key_result)))

    def test_flat_map(self):
        input_col = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(list(self.backend.flat_map(input_col, lambda x: x)),
                         [1, 2, 3, 4, 5, 6, 7, 8])

        input_col = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        self.assertEqual(list(self.backend.flat_map(input_col, lambda x: x[1])),
                         [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            list(
                self.backend.flat_map(input_col,
                                      lambda x: [(x[0], y) for y in x[1]])),
            [("a", 1), ("a", 2), ("a", 3), ("a", 4), ("b", 5), ("b", 6),
             ("b", 7), ("b", 8)])

    def test_group_by_key(self):
        some_dict = [("cheese", "brie"), ("bread", "sourdough"),
                     ("cheese", "swiss")]

        self.assertEqual(list(self.backend.group_by_key(some_dict)),
                         [("cheese", ["brie", "swiss"]),
                          ("bread", ["sourdough"])])

    def test_flatten(self):
        data1, data2, data3 = [1, 2, 3, 4], [5, 6, 7, 8], [9, 10]

        self.assertEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         list(self.backend.flatten((data1, data2, data3))))

    def test_distinct(self):
        input = [3, 2, 1, 3, 5, 4, 1, 1, 2]
        output = set(self.backend.distinct(input, "distinct"))
        self.assertSetEqual({1, 2, 3, 4, 5}, output)

    def test_output_reiterable(self):
        backend = pipeline_dp.LocalBackend()
        output = backend.map([1, 2, 3], lambda x: x, "Map")
        self.assertEqual(list(output), [1, 2, 3])
        self.assertEqual(list(output), [1, 2, 3])


class SumCombiner(dp_combiners.Combiner):

    def create_accumulator(self, values) -> float:
        return sum(values)

    def merge_accumulators(self, sum1: float, sum2: float):
        return sum1 + sum2

    def compute_metrics(self, sum: float) -> float:
        return sum

    def metrics_names(self) -> List[str]:
        return ['sum']

    def explain_computation(self) -> str:
        return "Compute non-dp Sum for tests"


if __name__ == '__main__':
    unittest.main()
