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
import random
import sys
import unittest
import warnings
from typing import List

from absl.testing import parameterized

from pipeline_dp import budget_accounting
from pipeline_dp import combiners as dp_combiners
from pipeline_dp import data_extractors
from pipeline_dp import spark_rdd_backend


@unittest.skipIf(sys.version_info.minor <= 7 and sys.version_info.major == 3,
                 "There are some problems with PySpark setup on older python.")
class SparkRDDBackendTest(parameterized.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore', ResourceWarning)
        import pyspark
        conf = pyspark.SparkConf().set("spark.log.level", "ERROR")
        cls.sc = pyspark.SparkContext.getOrCreate(conf=conf)
        cls.data_extractors = data_extractors.DataExtractors(
            partition_extractor=lambda x: x[1],
            privacy_id_extractor=lambda x: x[0],
            value_extractor=lambda x: x[2])
        cls.backend = spark_rdd_backend.SparkRDDBackend(cls.sc)

    def test_filter_by_key_none_keys_to_keep(self):
        data = [(1, 11), (2, 22)]
        dist_data = self.sc.parallelize(data)
        key_to_keep = None
        with self.assertRaises(TypeError):
            self.backend.filter_by_key(dist_data, key_to_keep)

    @parameterized.parameters({'distributed': False}, {'distributed': True})
    def test_filter_by_key_empty_keys_to_keep(self, distributed):
        data = [(1, 11), (2, 22)]
        dist_data = self.sc.parallelize(data)
        keys_to_keep = []
        if distributed:
            keys_to_keep = self.sc.parallelize(keys_to_keep)
        result = self.backend.filter_by_key(dist_data, keys_to_keep).collect()
        self.assertListEqual(result, [])

    @parameterized.parameters({'distributed': False}, {'distributed': True})
    def test_filter_by_key_nonempty_keys_to_keep(self, distributed):
        data = [(1, 11), (2, 22)]
        dist_data = self.sc.parallelize(data)
        keys_to_keep = [1, 3, 3]
        if distributed:
            keys_to_keep = self.sc.parallelize(keys_to_keep)
        result = self.backend.filter_by_key(dist_data, keys_to_keep).collect()
        self.assertListEqual(result, [(1, 11)])

    def test_sample_fixed_per_key(self):
        data = [(1, 11), (2, 22), (3, 33), (1, 14), (2, 25), (1, 16)]
        dist_data = self.sc.parallelize(data)
        rdd = self.backend.sample_fixed_per_key(dist_data, 2)
        result = dict(rdd.collect())
        self.assertEqual(len(result[1]), 2)
        self.assertTrue(set(result[1]).issubset({11, 14, 16}))
        self.assertSetEqual(set(result[2]), {22, 25})
        self.assertSetEqual(set(result[3]), {33})

    def test_count_per_element(self):
        data = ['a', 'b', 'a']
        dist_data = self.sc.parallelize(data)
        rdd = self.backend.count_per_element(dist_data)
        result = rdd.collect()
        result = dict(result)
        self.assertDictEqual(result, {'a': 2, 'b': 1})

    def test_sum_per_key(self):
        data = self.sc.parallelize([(1, 2), (2, 1), (1, 4), (3, 8), (2, -3),
                                    (10, 5)])
        result = self.backend.sum_per_key(data).collect()
        self.assertEqual(set(result), {(1, 6), (2, -2), (3, 8), (10, 5)})

    @unittest.skipIf(sys.platform == "darwin",
                     "There are some problems with PySpark setup on macOS")
    def test_combine_accumulators_per_key(self):
        data = self.sc.parallelize([(1, 2), (2, 1), (1, 4), (3, 8), (2, 3)])
        rdd = self.backend.group_by_key(data)
        sum_combiner = SumCombiner()
        rdd = self.backend.map_values(rdd, sum_combiner.create_accumulator)
        rdd = self.backend.combine_accumulators_per_key(rdd, sum_combiner)
        rdd = self.backend.map_values(rdd, sum_combiner.compute_metrics)
        result = dict(rdd.collect())
        self.assertDictEqual(result, {1: 6, 2: 4, 3: 8})

    def test_map_tuple(self):
        data = [(1, 2), (3, 4)]
        dist_data = self.sc.parallelize(data)
        result = self.backend.map_tuple(dist_data, lambda a, b: a + b).collect()
        self.assertEqual(result, [3, 7])

    def test_map_with_side_inputs(self):
        with self.assertRaises(NotImplementedError):
            self.backend.map_with_side_inputs(None, None, None,
                                              "map_with_side_inputs")

    def test_flat_map(self):
        data = [[1, 2, 3, 4], [5, 6, 7, 8]]
        dist_data = self.sc.parallelize(data)
        self.assertEqual(
            self.backend.flat_map(dist_data, lambda x: x).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])

        data = [("a", [1, 2, 3, 4]), ("b", [5, 6, 7, 8])]
        dist_data = self.sc.parallelize(data)
        self.assertEqual(
            self.backend.flat_map(dist_data, lambda x: x[1]).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            self.backend.flat_map(
                dist_data,
                lambda x: [(x[0], y) for y in x[1]]).collect(), [("a", 1),
                                                                 ("a", 2),
                                                                 ("a", 3),
                                                                 ("a", 4),
                                                                 ("b", 5),
                                                                 ("b", 6),
                                                                 ("b", 7),
                                                                 ("b", 8)])

    def test_flatten(self):
        data1 = self.sc.parallelize([1, 2, 3, 4])
        data2 = self.sc.parallelize([5, 6, 7, 8])

        self.assertEqual(
            self.backend.flatten((data1, data2)).collect(),
            [1, 2, 3, 4, 5, 6, 7, 8])

    def test_distinct(self):
        input = self.sc.parallelize([3, 2, 1, 3, 5, 4, 1, 1, 2])
        output = self.backend.distinct(input, "distinct").collect()
        self.assertSetEqual({1, 2, 3, 4, 5}, set(output))

    def test_cache_caches_rdd_and_is_idempotent(self):
        import pyspark
        rdd = self.backend.map(self.sc.parallelize(range(5)), lambda x:
                               (x, random.random()))
        cached_rdd = self.backend.cache(rdd)
        self.assertIs(cached_rdd, rdd)
        self.assertTrue(cached_rdd.is_cached)
        first = cached_rdd.collect()
        self.assertEqual(cached_rdd.collect(), first)
        # Calling cache() again must remain idempotent.
        self.assertIs(self.backend.cache(cached_rdd), cached_rdd)
        # Existing StorageLevel must not raise SparkUnsupportedOperationException.
        disk_rdd = self.sc.parallelize(range(5)).persist(
            pyspark.StorageLevel.MEMORY_AND_DISK)
        self.assertIs(self.backend.cache(disk_rdd), disk_rdd)
        self.assertEqual(disk_rdd.getStorageLevel(),
                         pyspark.StorageLevel.MEMORY_AND_DISK)

    def test_defer_cache_immediate_and_pre_budget_narrow_transforms(self):
        immediate = self.backend.map(self.sc.parallelize(range(5)), lambda x:
                                     (x, random.random()))
        self.assertIs(self.backend.defer_cache(immediate), immediate)
        self.assertTrue(immediate.is_cached)
        self.assertEqual(immediate.collect(), immediate.collect())

        accountant = budget_accounting.NaiveBudgetAccountant(1.0, 0.0)
        deferred = self.backend.map(self.sc.parallelize(range(5)), lambda x:
                                    (x, random.random()))
        self.assertIs(
            self.backend.defer_cache(deferred,
                                     accountant.run_after_compute_budgets),
            deferred)
        # Chain narrow transforms BEFORE compute_budgets().
        mapped = deferred.mapValues(lambda v: v * 2.0)
        chained = mapped.filter(lambda kv: kv[0] % 2 == 0).map(lambda kv:
                                                               (kv[0], kv[1]))
        self.assertFalse(deferred.is_cached)

        accountant.compute_budgets()
        self.assertTrue(deferred.is_cached)
        first_deferred = deferred.collect()
        first_mapped = mapped.collect()
        first_chained = chained.collect()
        for _ in range(5):
            self.assertEqual(deferred.collect(), first_deferred)
            self.assertEqual(mapped.collect(), first_mapped)
            self.assertEqual(chained.collect(), first_chained)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


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
