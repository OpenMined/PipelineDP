import unittest
import pipeline_dp
import typing
from unittest.mock import patch
from pipeline_dp import accumulator as acc
from pipeline_dp import aggregate_params as agg


class CompoundAccumulatorTest(unittest.TestCase):

    def test_with_mean_and_sum_squares(self):
        mean_acc = MeanAccumulator(params=[], values=[])
        sum_squares_acc = SumOfSquaresAccumulator(params=[], values=[])
        compound_accumulator = acc.CompoundAccumulator([mean_acc,
                                                      sum_squares_acc])

        compound_accumulator.add_value(3)
        compound_accumulator.add_value(4)

        computed_metrics = compound_accumulator.compute_metrics()
        self.assertTrue(isinstance(compound_accumulator, acc.CompoundAccumulator))
        self.assertEqual(len(computed_metrics), 2)
        self.assertEqual(computed_metrics, [3.5, 25])

    def test_adding_accumulator(self):
        mean_acc1 = MeanAccumulator(params=None, values=[5])
        sum_squares_acc1 = SumOfSquaresAccumulator(params=None, values=[5])
        compound_accumulator = acc.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])

        mean_acc2 = MeanAccumulator(params=[], values=[])
        sum_squares_acc2 = SumOfSquaresAccumulator(params=[], values=[])
        to_be_added_compound_accumulator = acc.CompoundAccumulator(
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

        base_compound_accumulator = acc.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])
        to_add_compound_accumulator = acc.CompoundAccumulator(
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

        base_compound_accumulator = acc.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])
        to_add_compound_accumulator = acc.CompoundAccumulator([mean_acc2])

        with self.assertRaises(ValueError) as context:
            base_compound_accumulator.add_accumulator(
                to_add_compound_accumulator)
        self.assertEqual(
            "Accumulators in the input are not of the same size. "
            "Expected size = 2 received size = 1.", str(context.exception))

    def test_serialization_single_accumulator(self):
        accumulator = MeanAccumulator(params=[], values=[5, 6])

        serialized_obj = accumulator.serialize()
        deserialized_obj =acc.Accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, MeanAccumulator)
        self.assertEqual(accumulator.sum, deserialized_obj.sum)
        self.assertEqual(accumulator.count, deserialized_obj.count)

    def test_serialization_compound_accumulator(self):
        mean_acc = MeanAccumulator(params=[], values=[15])
        sum_squares_acc = SumOfSquaresAccumulator(params=[], values=[1])
        compound_accumulator = acc.CompoundAccumulator([mean_acc, sum_squares_acc])

        serialized_obj = compound_accumulator.serialize()
        deserialized_obj = acc.Accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, acc.CompoundAccumulator)
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

        merged_accumulator = acc.merge([mean_accumulator1, mean_accumulator2])

        self.assertEqual(merged_accumulator.compute_metrics(), 10)

    def test_merge_diff_type_throws_type_error(self):
        mean_accumulator1 = MeanAccumulator(params=[], values=[15])
        sum_squares_acc = SumOfSquaresAccumulator(params=[], values=[1])

        with self.assertRaises(TypeError) as context:
            acc.merge([mean_accumulator1, sum_squares_acc])
        self.assertIn("The accumulator to be added is not of the same type."
                      "", str(context.exception))

    @patch('pipeline_dp.accumulator.create_accumulator_params')
    def test_accumulator_factory(self, mock_create_accumulator_params_function):
        aggregate_params = pipeline_dp.AggregateParams([
            agg.Metrics.MEAN], 5, 3)
        budget_accountant = pipeline_dp.BudgetAccountant(1, 0.01)

        values = [[10]]
        mock_create_accumulator_params_function.return_value = [
            acc.AccumulatorParams(MeanAccumulator, None)
        ]

        accumulator_factory = acc.AccumulatorFactory(aggregate_params,
                                                 budget_accountant)
        accumulator_factory.initialize()
        accumulator = accumulator_factory.create(values)

        self.assertTrue(isinstance(accumulator, MeanAccumulator))
        self.assertEqual(accumulator.compute_metrics(), 10)

    @patch('pipeline_dp.accumulator.create_accumulator_params')
    def test_accumulator_factory_multiple_types(
            self, mock_create_accumulator_params_function):
        aggregate_params = pipeline_dp.AggregateParams(
            [agg.Metrics.MEAN, agg.Metrics.VAR], 5, 3)
        budget_accountant = pipeline_dp.BudgetAccountant(1, 0.01)
        values = [[10], [10]]

        mock_create_accumulator_params_function.return_value = [
            acc.AccumulatorParams(MeanAccumulator, None),
            acc.AccumulatorParams(SumOfSquaresAccumulator, None)
        ]

        accumulator_factory = acc.AccumulatorFactory(aggregate_params,
                                                 budget_accountant)
        accumulator_factory.initialize()
        accumulator = accumulator_factory.create(values)

        self.assertTrue(isinstance(accumulator, acc.CompoundAccumulator))
        self.assertEqual(accumulator.compute_metrics(), [10, 100])


class MeanAccumulator(acc.Accumulator):

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
        if not isinstance(accumulator, MeanAccumulator):
            raise TypeError(
                "The accumulator to be added is not of the same type.")
        self.sum += accumulator.sum
        self.count += accumulator.count
        return self

    def compute_metrics(self):
        if self.count == 0:
            return float('NaN')
        return self.sum / self.count


# Accumulator classes for testing
class SumOfSquaresAccumulator(acc.Accumulator):

    def __init__(self, params, values: typing.Iterable[float] = []):
        self.sum_squares = sum([value * value for value in values])
        self.params = params

    def add_value(self, v):
        self.sum_squares += v * v
        return self

    def add_accumulator(
            self, accumulator: 'SumOfSquaresAccumulator'
    ) -> 'SumOfSquaresAccumulator':
        if not isinstance(accumulator, SumOfSquaresAccumulator):
            raise TypeError(
                "The accumulator to be added is not of the same type.")
        self.sum_squares += accumulator.sum_squares
        return self

    def compute_metrics(self):
        return self.sum_squares


if __name__ == '__main__':
    unittest.main()
