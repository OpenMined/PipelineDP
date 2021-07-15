import unittest
import pipeline_dp
import typing
import numpy as np
import pipeline_dp.accumulator as accumulator


class CompoundAccumulatorTest(unittest.TestCase):

    def test_with_mean_and_sum_squares(self):
        mean_acc = MeanAccumulator()
        sum_squares_acc = SumOfSquaresAccumulator()
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
        mean_acc1 = MeanAccumulator().add_value(5)
        sum_squares_acc1 = SumOfSquaresAccumulator().add_value(5)
        compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])

        mean_acc2 = MeanAccumulator()
        sum_squares_acc2 = SumOfSquaresAccumulator()
        to_be_added_compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc2, sum_squares_acc2])

        to_be_added_compound_accumulator.add_value(4)

        compound_accumulator.add_accumulator(to_be_added_compound_accumulator)
        compound_accumulator.add_value(3)

        computed_metrics = compound_accumulator.compute_metrics()
        self.assertEqual(len(computed_metrics), 2)
        self.assertEqual(computed_metrics, [4, 50])

    def test_adding_mismatched_accumulator_order_raises_exception(self):
        mean_acc1 = MeanAccumulator().add_value(11)
        sum_squares_acc1 = SumOfSquaresAccumulator().add_value(1)
        mean_acc2 = MeanAccumulator().add_value(22)
        sum_squares_acc2 = SumOfSquaresAccumulator().add_value(2)

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
        mean_acc1 = MeanAccumulator().add_value(11)
        sum_squares_acc1 = SumOfSquaresAccumulator().add_value(1)
        mean_acc2 = MeanAccumulator().add_value(22)

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
        accumulator = MeanAccumulator().add_value(5).add_value(6)

        serialized_obj = accumulator.serialize()
        deserialized_obj = accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, MeanAccumulator)
        self.assertEqual(accumulator.sum, deserialized_obj.sum)
        self.assertEqual(accumulator.count, deserialized_obj.count)

    def test_serialization_compound_accumulator(self):
        mean_acc = MeanAccumulator().add_value(15)
        sum_squares_acc = SumOfSquaresAccumulator().add_value(1)
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
        mean_accumulator = MeanAccumulator().add_value(15)

        serialized_obj = mean_accumulator.serialize()

        with self.assertRaises(TypeError) as context:
            SumOfSquaresAccumulator.deserialize(serialized_obj)
        self.assertEqual("The deserialized object is not of the right type.",
                         str(context.exception))


class GenericAccumulatorTest(unittest.TestCase):

    def test_merge_accumulators(self):
        mean_accumulator1 = MeanAccumulator().add_value(15)
        mean_accumulator2 = MeanAccumulator().add_value(5)

        merged_accumulator = accumulator.merge(
            [mean_accumulator1, mean_accumulator2])

        self.assertEqual(merged_accumulator.compute_metrics(), 10)

    def test_merge_diff_type_throws_type_error(self):
        mean_accumulator1 = MeanAccumulator().add_value(15)
        sum_squares_acc = SumOfSquaresAccumulator().add_value(1)

        with self.assertRaises(TypeError) as context:
            accumulator.merge([mean_accumulator1, sum_squares_acc])
        self.assertIn("The accumulator to be added is not of the same type."
                      "", str(context.exception))


class MeanAccumulator(accumulator.Accumulator):

    def __init__(self, accumulators: typing.Iterable['MeanAccumulator'] = None):
        self.sum = np.sum([concat_acc.sum for concat_acc in accumulators
                          ]) if accumulators else 0
        self.count = np.sum([concat_acc.count for concat_acc in accumulators
                            ]) if accumulators else 0

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
class SumOfSquaresAccumulator(accumulator.Accumulator):

    def __init__(
            self,
            accumulators: typing.Iterable['SumOfSquaresAccumulator'] = None):
        self.sum_squares = np.sum([
            concat_acc.sum_squares for concat_acc in accumulators
        ]) if accumulators else 0

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


class CountAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(), list(range(5)))
        assert count_accumulator.compute_metrics() == 5

        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(), 'a' * 50)
        assert count_accumulator.compute_metrics() == 50

        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(), list(range(50)))
        count_accumulator.add_value(49)
        assert count_accumulator.compute_metrics() == 99

        count_accumulator_1 = accumulator.CountAccumulator(
            accumulator.CountParams(), list(range(50)))
        count_accumulator_2 = accumulator.CountAccumulator(
            accumulator.CountParams(), 'a' * 50)
        count_accumulator_1.add_accumulator(count_accumulator_2)
        assert count_accumulator_1.compute_metrics() == 100


if __name__ == '__main__':
    unittest.main()
