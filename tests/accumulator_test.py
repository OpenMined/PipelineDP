import unittest
import typing
from unittest.mock import patch

import numpy as np
import pipeline_dp
from pipeline_dp import aggregate_params as agg
from pipeline_dp.dp_computations import MeanVarParams
import pipeline_dp.accumulator as accumulator


class CompoundAccumulatorTest(unittest.TestCase):

    def test_with_mean_and_sum_squares(self):
        mean_acc = MeanAccumulator(params=[], values=[])
        sum_squares_acc = SumOfSquaresAccumulator(params=[], values=[])
        compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc, sum_squares_acc])

        compound_accumulator.add_value(3)
        compound_accumulator.add_value(4)

        computed_metrics = compound_accumulator.compute_metrics()
        self.assertTrue(
            isinstance(compound_accumulator, accumulator.CompoundAccumulator))
        self.assertEqual(len(computed_metrics), 2)
        self.assertEqual(computed_metrics, [3.5, 25])

    def test_adding_accumulator(self):
        mean_acc1 = MeanAccumulator(params=None, values=[5])
        sum_squares_acc1 = SumOfSquaresAccumulator(params=None, values=[5])
        compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])

        mean_acc2 = MeanAccumulator(params=[], values=[])
        sum_squares_acc2 = SumOfSquaresAccumulator(params=[], values=[])
        to_be_added_compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc2, sum_squares_acc2])

        to_be_added_compound_accumulator.add_value(4)

        compound_accumulator.add_accumulator(to_be_added_compound_accumulator)
        compound_accumulator.add_value(3)

        computed_metrics = compound_accumulator.compute_metrics()
        self.assertEqual(len(computed_metrics), 2)
        self.assertEqual(computed_metrics, [4, 50])

    def test_adding_mismatched_accumulator_order_raises_exception(self):
        mean_acc1 = MeanAccumulator(params=[], values=[11])
        sum_squares_acc1 = SumOfSquaresAccumulator(params=[], values=[1])
        mean_acc2 = MeanAccumulator(params=[], values=[22])
        sum_squares_acc2 = SumOfSquaresAccumulator(params=[], values=[2])

        base_compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])
        to_add_compound_accumulator = accumulator.CompoundAccumulator(
            [sum_squares_acc2, mean_acc2])

        with self.assertRaises(TypeError) as context:
            base_compound_accumulator.add_accumulator(
                to_add_compound_accumulator)
        self.assertEqual(
            "The type of the accumulators don't match at index 0. "
            "MeanAccumulator != SumOfSquaresAccumulator."
            "", str(context.exception))

    def test_adding_mismatched_accumulator_length_raises_exception(self):
        mean_acc1 = MeanAccumulator(params=[], values=[11])
        sum_squares_acc1 = SumOfSquaresAccumulator(params=[], values=[1])
        mean_acc2 = MeanAccumulator(params=[], values=[22])

        base_compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])
        to_add_compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc2])

        with self.assertRaises(ValueError) as context:
            base_compound_accumulator.add_accumulator(
                to_add_compound_accumulator)
        self.assertEqual(
            "Accumulators in the input are not of the same size. "
            "Expected size = 2 received size = 1.", str(context.exception))

    def test_serialization_single_accumulator(self):
        mean_acc = MeanAccumulator(params=[], values=[5, 6])

        serialized_obj = mean_acc.serialize()
        deserialized_obj = accumulator.Accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, MeanAccumulator)
        self.assertEqual(mean_acc.sum, deserialized_obj.sum)
        self.assertEqual(mean_acc.count, deserialized_obj.count)

    def test_serialization_compound_accumulator(self):
        mean_acc = MeanAccumulator(params=[], values=[15])
        sum_squares_acc = SumOfSquaresAccumulator(params=[], values=[1])
        compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc, sum_squares_acc])

        serialized_obj = compound_accumulator.serialize()
        deserialized_obj = accumulator.Accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, accumulator.CompoundAccumulator)

        self.assertEqual(len(deserialized_obj.accumulators), 2)
        self.assertIsInstance(deserialized_obj.accumulators[0], MeanAccumulator)
        self.assertIsInstance(deserialized_obj.accumulators[1],
                              SumOfSquaresAccumulator)
        self.assertEqual(deserialized_obj.compute_metrics(),
                         compound_accumulator.compute_metrics())

    def test_serialization_with_incompatible_serialized_object(self):
        mean_accumulator = MeanAccumulator(params=[], values=[15])

        serialized_obj = mean_accumulator.serialize()

        with self.assertRaises(TypeError) as context:
            SumOfSquaresAccumulator.deserialize(serialized_obj)
        self.assertEqual("The deserialized object is not of the right type.",
                         str(context.exception))


class GenericAccumulatorTest(unittest.TestCase):

    def test_merge_accumulators(self):
        mean_accumulator1 = MeanAccumulator(params=[], values=[15])
        mean_accumulator2 = MeanAccumulator(params=[], values=[5])

        merged_accumulator = accumulator.merge(
            [mean_accumulator1, mean_accumulator2])

        self.assertEqual(merged_accumulator.compute_metrics(), 10)

        vec_sum_accumulator1 = accumulator.VectorSummationAccumulator(
            params=None, values=[(15, 2)])
        vec_sum_accumulator2 = accumulator.VectorSummationAccumulator(
            params=None, values=[(27, 40)])
        merged_accumulator = accumulator.merge(
            [vec_sum_accumulator1, vec_sum_accumulator2])
        self.assertEqual(tuple(merged_accumulator.compute_metrics()), (42, 42))

    def test_merge_diff_type_throws_type_error(self):
        mean_accumulator1 = MeanAccumulator(params=[], values=[15])
        sum_squares_acc = SumOfSquaresAccumulator(params=[], values=[1])
        vec_sum_accumulator = accumulator.VectorSummationAccumulator(
            params=accumulator.VectorSummationParams(agg.NoiseKind.LAPLACE,
                                                     float('inf')),
            values=[(27, 40)])

        with self.assertRaises(TypeError) as context:
            accumulator.merge([mean_accumulator1, sum_squares_acc])

        self.assertIn("The accumulator to be added is not of the same type"
                      "", str(context.exception))

        with self.assertRaises(TypeError) as context:
            accumulator.merge([vec_sum_accumulator, sum_squares_acc])

        self.assertIn("The accumulator to be added is not of the same type"
                      "", str(context.exception))

    @patch('pipeline_dp.accumulator.create_accumulator_params')
    def test_accumulator_factory(self, mock_create_accumulator_params_function):
        aggregate_params = pipeline_dp.AggregateParams([agg.Metrics.MEAN], 5, 3)
        budget_accountant = pipeline_dp.BudgetAccountant(1, 0.01)

        values = [10]
        mock_create_accumulator_params_function.return_value = [
            accumulator.AccumulatorParams(MeanAccumulator, None)
        ]

        accumulator_factory = accumulator.AccumulatorFactory(
            aggregate_params, budget_accountant)
        accumulator_factory.initialize()
        created_accumulator = accumulator_factory.create(values)

        self.assertTrue(isinstance(created_accumulator, MeanAccumulator))
        self.assertEqual(created_accumulator.compute_metrics(), 10)
        mock_create_accumulator_params_function.assert_called_with(
            aggregate_params, budget_accountant)

    @patch('pipeline_dp.accumulator.create_accumulator_params')
    def test_accumulator_factory_multiple_types(
            self, mock_create_accumulator_params_function):
        aggregate_params = pipeline_dp.AggregateParams(
            [agg.Metrics.MEAN, agg.Metrics.VAR], 5, 3)
        budget_accountant = pipeline_dp.BudgetAccountant(1, 0.01)
        values = [10]

        mock_create_accumulator_params_function.return_value = [
            accumulator.AccumulatorParams(MeanAccumulator, None),
            accumulator.AccumulatorParams(SumOfSquaresAccumulator, None)
        ]

        accumulator_factory = accumulator.AccumulatorFactory(
            aggregate_params, budget_accountant)
        accumulator_factory.initialize()
        created_accumulator = accumulator_factory.create(values)

        self.assertTrue(
            isinstance(created_accumulator, accumulator.CompoundAccumulator))
        self.assertEqual(created_accumulator.compute_metrics(), [10, 100])
        mock_create_accumulator_params_function.assert_called_with(
            aggregate_params, budget_accountant)

    def test_create_accumulator_params_with_count_params(self):
        acc_params = accumulator.create_accumulator_params(
            aggregation_params=pipeline_dp.AggregateParams(
                metrics=[pipeline_dp.Metrics.COUNT],
                max_partitions_contributed=4,
                max_contributions_per_partition=5,
                budget_weight=1),
            budget_accountant=pipeline_dp.BudgetAccountant(1, 0.01))
        self.assertEqual(len(acc_params), 1)
        self.assertEqual(acc_params[0].accumulator_type,
                         accumulator.CountAccumulator)
        self.assertTrue(
            isinstance(acc_params[0].constructor_params,
                       accumulator.CountParams))


class MeanAccumulator(accumulator.Accumulator):

    def __init__(self, params, values: typing.Iterable[float] = []):
        self.sum = sum(values)
        self.count = len(values)
        self.params = params

    def add_value(self, v):
        self.sum += v
        self.count += 1
        return self

    def add_accumulator(self,
                        accumulator: 'MeanAccumulator') -> 'MeanAccumulator':
        self._check_mergeable(accumulator)
        self.sum += accumulator.sum
        self.count += accumulator.count
        return self

    def compute_metrics(self):
        if self.count == 0:
            return float('NaN')
        return self.sum / self.count


# Accumulator classes for testing
class SumOfSquaresAccumulator(accumulator.Accumulator):

    def __init__(self, params, values: typing.Iterable[float] = []):
        self.sum_squares = sum([value * value for value in values])
        self.params = params

    def add_value(self, v):
        self.sum_squares += v * v
        return self

    def add_accumulator(
            self, accumulator: 'SumOfSquaresAccumulator'
    ) -> 'SumOfSquaresAccumulator':
        self._check_mergeable(accumulator)
        self.sum_squares += accumulator.sum_squares
        return self

    def compute_metrics(self):
        return self.sum_squares


class CountAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(), list(range(5)))
        self.assertEqual(count_accumulator.compute_metrics(), 5)

        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(), 'a' * 50)
        self.assertEqual(count_accumulator.compute_metrics(), 50)

        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(), list(range(50)))
        count_accumulator.add_value(49)
        self.assertEqual(count_accumulator.compute_metrics(), 51)

        count_accumulator_1 = accumulator.CountAccumulator(
            accumulator.CountParams(), list(range(50)))
        count_accumulator_2 = accumulator.CountAccumulator(
            accumulator.CountParams(), 'a' * 50)
        count_accumulator_1.add_accumulator(count_accumulator_2)
        self.assertEqual(count_accumulator_1.compute_metrics(), 100)


class SumAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        no_noise = MeanVarParams(eps=1,
                                 delta=1,
                                 low=0,
                                 high=0,
                                 max_partitions_contributed=1,
                                 max_contributions_per_partition=1,
                                 noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
        sum_accumulator = accumulator.SumAccumulator(
            accumulator.SumParams(no_noise), list(range(6)))

        self.assertEqual(sum_accumulator.compute_metrics(), 15)

        sum_accumulator.add_value(5)
        self.assertEqual(sum_accumulator.compute_metrics(), 20)

        sum_accumulator_2 = accumulator.SumAccumulator(
            accumulator.SumParams(no_noise), list(range(3)))

        sum_accumulator.add_accumulator(sum_accumulator_2)
        self.assertEqual(sum_accumulator.compute_metrics(), 23)

    def test_with_noise(self):
        sum_accumulator = accumulator.SumAccumulator(
            accumulator.SumParams(
                MeanVarParams(eps=10,
                              delta=1e-5,
                              low=0,
                              high=1,
                              max_partitions_contributed=1,
                              max_contributions_per_partition=3,
                              noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
            list(range(6)))
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=15,
                               delta=4)

        sum_accumulator.add_value(5)
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=20,
                               delta=4)

        sum_accumulator_2 = accumulator.SumAccumulator(
            accumulator.SumParams(
                MeanVarParams(eps=10,
                              delta=1e-5,
                              low=0,
                              high=1,
                              max_partitions_contributed=1,
                              max_contributions_per_partition=3,
                              noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)),
            list(range(3)))
        sum_accumulator.add_accumulator(sum_accumulator_2)
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=23,
                               delta=4)


class VectorSummuationAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        vec_sum_accumulator = accumulator.VectorSummationAccumulator(
            params=None, values=[(1, 2), (3, 4), (5, 6)])
        self.assertEqual(tuple(vec_sum_accumulator.compute_metrics()), (9, 12))

        vec_sum_accumulator.add_value((7, 8))
        self.assertTrue(
            np.all(vec_sum_accumulator.compute_metrics() == np.array([16, 20])))

        vec_sum_accumulator_2 = accumulator.VectorSummationAccumulator(
            params=None, values=[(1, 2), (1, 4), (1, 8), (1, 16)])
        vec_sum_accumulator.add_accumulator(vec_sum_accumulator_2)
        self.assertEqual(tuple(vec_sum_accumulator.compute_metrics()), (20, 50))


if __name__ == '__main__':
    unittest.main()
