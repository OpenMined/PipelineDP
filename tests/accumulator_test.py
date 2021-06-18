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

  def test_with_compound_accumulator(self):
    mean_acc = MeanAccumulator().add_value(11)
    sum_squares_acc = SumOfSquaresAccumulator().add_value(1)
    compound_accumulator1 = pipeline_dp.CompoundAccumulator.merge(
      [mean_acc])
    compound_accumulator2 = pipeline_dp.CompoundAccumulator.merge(
      [sum_squares_acc])

    compound_accumulator1.add_value(1)
    compound_accumulator2.add_value(2)

    merged_compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [compound_accumulator1, compound_accumulator2])
    merged_compound_accumulator.add_value(3);

    computed_metrics = merged_compound_accumulator.compute_metrics()
    self.assertEqual(len(computed_metrics), 2)
    self.assertEqual(computed_metrics, [5, 14])

  def test_adding_accumulator(self):
    compound_accumulator = pipeline_dp.CompoundAccumulator()

    mean_acc = MeanAccumulator()
    sum_squares_acc = SumOfSquaresAccumulator()
    to_be_added_compound_accumulator = pipeline_dp.CompoundAccumulator.merge(
      [mean_acc, sum_squares_acc])

    to_be_added_compound_accumulator.add_value(4)

    compound_accumulator.add_accumulator(to_be_added_compound_accumulator)
    to_be_added_compound_accumulator.add_value(5)

    computed_metrics = compound_accumulator.compute_metrics()
    self.assertEqual(len(computed_metrics), 2)
    self.assertEqual(computed_metrics, [4.5, 41])


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
