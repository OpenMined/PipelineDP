import unittest
import pipeline_dp
import typing
import numpy as np


class CompoundAccumulatorTest(unittest.TestCase):

  def test_with_mean_and_sum_squares(self):
    mean_acc = MeanAccumulator()
    sum_squares_acc = SumOfSquaresAccumulator()
    compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [mean_acc, sum_squares_acc])

    compound_accumulator.add_value(3)
    compound_accumulator.add_value(4)

    computed_metrics = compound_accumulator.compute_metrics()
    self.assertTrue(
      isinstance(compound_accumulator, pipeline_dp.CompoundAccumulator))
    self.assertEqual(len(computed_metrics), 2)
    self.assertEqual(computed_metrics, [3.5, 25])

  def test_adding_accumulator(self):
    mean_acc1 = MeanAccumulator().add_value(5)
    sum_squares_acc1 = SumOfSquaresAccumulator().add_value(5)
    compound_accumulator = pipeline_dp.CompoundAccumulator.merge([mean_acc1,
                                                                  sum_squares_acc1])

    mean_acc2 = MeanAccumulator()
    sum_squares_acc2 = SumOfSquaresAccumulator()
    to_be_added_compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [mean_acc2, sum_squares_acc2])

    to_be_added_compound_accumulator.add_value(4)

    compound_accumulator.add_accumulator(to_be_added_compound_accumulator)
    compound_accumulator.add_value(3)

    computed_metrics = compound_accumulator.compute_metrics()
    self.assertEqual(len(computed_metrics), 2)
    self.assertEqual(computed_metrics, [4, 50])

  def test_adding_mismatched_accumulator_raises_exception(self):
    mean_acc = MeanAccumulator().add_value(11)
    sum_squares_acc = SumOfSquaresAccumulator().add_value(1)

    base_compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [mean_acc])
    to_add_compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [sum_squares_acc])

    with self.assertRaises(ValueError) as context:
      base_compound_accumulator.add_accumulator(to_add_compound_accumulator)
    self.assertEqual("Accumulators in the input are not of the same size "
                     + "or don't match the type/order of the base accumulators.",
                     str(context.exception))

  def test_serialization_single_accumulator(self):
    accumulator = MeanAccumulator().add_value(5).add_value(6)

    serialized_obj = accumulator.serialize()
    deserialized_obj = pipeline_dp.Accumulator.deserialize(serialized_obj)

    self.assertIsInstance(deserialized_obj, MeanAccumulator)
    self.assertEqual(accumulator.sum, deserialized_obj.sum)
    self.assertEqual(accumulator.count, deserialized_obj.count)

  def test_serialization_compound_accumulator(self):
    mean_acc = MeanAccumulator().add_value(15)
    sum_squares_acc = SumOfSquaresAccumulator().add_value(1)
    compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [mean_acc, sum_squares_acc])

    serialized_obj = compound_accumulator.serialize()
    deserialized_obj = pipeline_dp.Accumulator.deserialize(serialized_obj)

    self.assertIsInstance(deserialized_obj, pipeline_dp.CompoundAccumulator)
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
    self.assertEqual("The deserialized object is not of the right type",
                     str(context.exception))


class MeanAccumulator(pipeline_dp.Accumulator):

  def __init__(self, accumulators: typing.Iterable[
    'MeanAccumulator'] = None):
    self.sum = np.sum([concat_acc.sum
                       for concat_acc in accumulators]) if accumulators else 0
    self.count = np.sum([concat_acc.count
                         for concat_acc in accumulators]) if accumulators else 0

  def add_value(self, v):
    self.sum += v
    self.count += 1
    return self

  def add_accumulator(self,
                      accumulator: 'MeanAccumulator') -> 'MeanAccumulator':
    self.sum += accumulator.sum
    self.count += accumulator.count
    return self

  def compute_metrics(self):
    if self.count == 0:
      return float('NaN')
    return self.sum / self.count


# Accumulator classes for testing
class SumOfSquaresAccumulator(pipeline_dp.Accumulator):

  def __init__(self, accumulators: typing.Iterable[
    'MeanAccumulator'] = None):
    self.sum_squares = np.sum([concat_acc.sum_squares
                               for concat_acc in
                               accumulators]) if accumulators else 0

  def add_value(self, v):
    self.sum_squares += v * v
    return self

  def add_accumulator(self,
                      accumulator: 'MeanAccumulator') -> 'MeanAccumulator':
    self.sum_squares += accumulator.sum_squares
    return self

  def compute_metrics(self):
    return self.sum_squares


if __name__ == '__main__':
  unittest.main()
