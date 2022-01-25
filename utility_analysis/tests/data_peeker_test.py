"""Tests for data_peeker."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
import pipeline_dp

from utility_analysis import data_peeker


class DataPeekerTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        pipeline_backend = pipeline_dp.LocalBackend()
        self._peeker = data_peeker.DataPeeker(pipeline_backend)

    @parameterized.named_parameters(
        dict(testcase_name='sum', metric=pipeline_dp.Metrics.SUM),
        dict(testcase_name='count', metric=pipeline_dp.Metrics.COUNT),
    )
    def test_sketch(self, metric):
        # Create dataset.
        number_of_sampled_partitions = 5
        input_size = 100
        input_data = range(input_size)

        # Run sketch and check results
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f'pid{x}',
            partition_extractor=lambda x: f'pk{x%10}',
            value_extractor=lambda x: x)
        params = data_peeker.SampleParams(
            number_of_sampled_partitions=number_of_sampled_partitions,
            metrics=[metric])
        key_to_pv_pcount = collections.defaultdict(list)
        for pk, pv, pcount in self._peeker.sketch(input_data, params,
                                                  data_extractor):
            key_to_pv_pcount[pk].append((pv, pcount))
        self.assertLen(key_to_pv_pcount, number_of_sampled_partitions)
        for pv_pcount_list in key_to_pv_pcount.values():
            self.assertLen(pv_pcount_list, input_size / 10)
            self.assertTrue(all(pcount == 1 for _, pcount in pv_pcount_list))
            if metric == pipeline_dp.Metrics.COUNT:
                self.assertTrue(all(pval == 1 for pval, _ in pv_pcount_list))

    def test_sample_size(self):
        # Create dataset.
        number_of_sampled_partitions = 5
        input_size = 100
        input_data = range(input_size)

        # Run sample and check results
        params = data_peeker.SampleParams(
            number_of_sampled_partitions=number_of_sampled_partitions)
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f'pid{x}',
            partition_extractor=lambda x: f'pk{x%10}',
            value_extractor=lambda x: x)
        key_to_v = collections.defaultdict(list)
        for _, pk, pv in self._peeker.sample(input_data, params,
                                             data_extractor):
            key_to_v[pk].append(pv)
        self.assertLen(key_to_v, number_of_sampled_partitions)
        for v_list in key_to_v.values():
            self.assertLen(v_list, input_size / 10)

    def test_aggregate_true(self):
        input_data = range(100)
        params = data_peeker.SampleParams(number_of_sampled_partitions=10,
                                          metrics=[pipeline_dp.Metrics.COUNT])
        data_extractor = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda x: f'pid{x}',
            partition_extractor=lambda x: f'pk{x%10}',
            value_extractor=lambda x: x)
        results = self._peeker.aggregate_true(input_data, params,
                                              data_extractor)
        key_to_v = collections.defaultdict(list)
        for pk, pv in results:
            key_to_v[pk].append(pv)
            self.assertEqual(pv[0], 10)
        self.assertLen(key_to_v.keys(), 10)


if __name__ == '__main__':
    absltest.main()
