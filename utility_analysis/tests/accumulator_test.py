"""Tests for accumulator."""

from absl.testing import absltest
from absl.testing import parameterized
import pipeline_dp

from utility_analysis import accumulator


class AccumulatorTest(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name='one', data=range(1)),
        dict(testcase_name='two', data=range(2)),
        dict(testcase_name='hundred', data=range(100)),
    )
    def test_count(self, data):
        count_accumulator = accumulator.CountAccumulator(data)
        self.assertEqual(count_accumulator.compute_metrics(), len(data))
        count_accumulator.add_value(1)
        self.assertEqual(count_accumulator.compute_metrics(), len(data) + 1)

    @parameterized.named_parameters(
        dict(testcase_name='one_merge', first=range(1)),
        dict(testcase_name='two_merge', first=range(2), second=range(1)),
        dict(testcase_name='three_merge',
             first=range(100),
             second=range(1),
             third=range(3)),
    )
    def test_count_merge(self, **kwargs):
        count_accumulator = accumulator.CountAccumulator([])
        total_count = 0
        for data in kwargs.values():
            total_count += len(data)
            count_accumulator.add_accumulator(
                accumulator.CountAccumulator(data))
        self.assertEqual(count_accumulator.compute_metrics(), total_count)

    @parameterized.named_parameters(
        dict(testcase_name='one', data=range(1), delta=2),
        dict(testcase_name='two', data=range(2), delta=3),
        dict(testcase_name='hundred', data=range(100), delta=4),
    )
    def test_sum(self, data, delta):
        sum_accumulator = accumulator.SumAccumulator(data)
        self.assertEqual(sum_accumulator.compute_metrics(), sum(data))
        sum_accumulator.add_value(delta)
        self.assertEqual(sum_accumulator.compute_metrics(), sum(data) + delta)

    @parameterized.named_parameters(
        dict(testcase_name='one_merge', first=range(1)),
        dict(testcase_name='two_merge', first=range(2), second=range(1)),
        dict(testcase_name='three_merge',
             first=range(100),
             second=range(1),
             third=range(3)),
    )
    def test_sum_merge(self, **kwargs):
        sum_accumulator = accumulator.SumAccumulator([])
        total_sum = 0
        for data in kwargs.values():
            total_sum += sum(data)
            sum_accumulator.add_accumulator(accumulator.SumAccumulator(data))
        self.assertEqual(sum_accumulator.compute_metrics(), total_sum)

    @parameterized.named_parameters(
        dict(testcase_name='one', data=range(1)),
        dict(testcase_name='two', data=range(2)),
        dict(testcase_name='hundred', data=range(100)),
    )
    def test_privacy_id_count(self, data):
        id_count_accumulator = accumulator.PrivacyIdCountAccumulator(data)
        self.assertEqual(id_count_accumulator.compute_metrics(), 1)

    @parameterized.named_parameters(
        dict(testcase_name='one_merge', first=range(1)),
        dict(testcase_name='two_merge', first=range(2), second=range(1)),
        dict(testcase_name='three_merge',
             first=range(100),
             second=range(1),
             third=range(3)),
    )
    def test_id_count_merge(self, **kwargs):
        id_count_accumulator = accumulator.PrivacyIdCountAccumulator([])
        total_id_count = len(kwargs) + 1
        for data in kwargs.values():
            id_count_accumulator.add_accumulator(
                accumulator.PrivacyIdCountAccumulator(data))
        self.assertEqual(id_count_accumulator.compute_metrics(), total_id_count)

    @parameterized.named_parameters(
        dict(testcase_name='sum',
             data=[1, 2, 3],
             metrics=[pipeline_dp.Metrics.SUM],
             result=[6]),
        dict(testcase_name='count',
             data=[1, 2, 3],
             metrics=[pipeline_dp.Metrics.COUNT],
             result=[3]),
        dict(testcase_name='sum_and_count',
             data=[1, 2, 3],
             metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
             result=[3, 6]),
        dict(testcase_name='count_and_id_count',
             data=[1, 2, 3],
             metrics=[
                 pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT
             ],
             result=[3, 1]),
        dict(testcase_name='sum_count_and_id_count',
             data=[1, 2],
             metrics=[
                 pipeline_dp.Metrics.SUM, pipeline_dp.Metrics.COUNT,
                 pipeline_dp.Metrics.PRIVACY_ID_COUNT
             ],
             result=[2, 3, 1]),
    )
    def test_accumulator_factory(self, data, metrics, result):
        accumulator_factory = accumulator.CompoundAccumulatorFactory(metrics)
        created_accumulator = accumulator_factory.create(data)
        self.assertIsInstance(created_accumulator,
                              pipeline_dp.accumulator.CompoundAccumulator)
        # Results are orderred by COUNT, SUM, ID_COUNT
        self.assertEqual(created_accumulator.compute_metrics(), result)


if __name__ == '__main__':
    absltest.main()
