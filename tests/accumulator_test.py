from pipeline_dp import dp_computations
import unittest
import typing
from unittest.mock import patch

import numpy as np
import pipeline_dp
from pipeline_dp import aggregate_params as agg
from pipeline_dp.budget_accounting import NaiveBudgetAccountant
from pipeline_dp.aggregate_params import NoiseKind
from pipeline_dp.aggregate_params import NormKind
import pipeline_dp.accumulator as accumulator


class CompoundAccumulatorTest(unittest.TestCase):

    def test_with_mean_and_sum_squares(self):
        mean_acc = MeanAccumulator(values=[])
        sum_squares_acc = SumOfSquaresAccumulator(values=[])
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
        mean_acc1 = MeanAccumulator(values=[5])
        sum_squares_acc1 = SumOfSquaresAccumulator(values=[5])
        compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc1, sum_squares_acc1])

        mean_acc2 = MeanAccumulator(values=[])
        sum_squares_acc2 = SumOfSquaresAccumulator(values=[])
        to_be_added_compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc2, sum_squares_acc2])

        to_be_added_compound_accumulator.add_value(4)

        compound_accumulator.add_accumulator(to_be_added_compound_accumulator)
        compound_accumulator.add_value(3)

        computed_metrics = compound_accumulator.compute_metrics()
        self.assertEqual(len(computed_metrics), 2)
        self.assertEqual(computed_metrics, [4, 50])

    def test_adding_mismatched_accumulator_order_raises_exception(self):
        mean_acc1 = MeanAccumulator(values=[11])
        sum_squares_acc1 = SumOfSquaresAccumulator(values=[1])
        mean_acc2 = MeanAccumulator(values=[22])
        sum_squares_acc2 = SumOfSquaresAccumulator(values=[2])

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
        mean_acc1 = MeanAccumulator(values=[11])
        sum_squares_acc1 = SumOfSquaresAccumulator(values=[1])
        mean_acc2 = MeanAccumulator(values=[22])

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
        mean_acc = MeanAccumulator(values=[5, 6])

        serialized_obj = mean_acc.serialize()
        deserialized_obj = accumulator.Accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, MeanAccumulator)
        self.assertEqual(mean_acc.sum, deserialized_obj.sum)
        self.assertEqual(mean_acc.count, deserialized_obj.count)

    def test_serialization_compound_accumulator(self):
        mean_acc = MeanAccumulator(values=[15])
        sum_squares_acc = SumOfSquaresAccumulator(values=[1])
        compound_accumulator = accumulator.CompoundAccumulator(
            [mean_acc, sum_squares_acc])

        serialized_obj = compound_accumulator.serialize()
        deserialized_obj = accumulator.Accumulator.deserialize(serialized_obj)

        self.assertIsInstance(deserialized_obj, accumulator.CompoundAccumulator)

        self.assertEqual(len(deserialized_obj._accumulators), 2)
        self.assertIsInstance(deserialized_obj._accumulators[0],
                              MeanAccumulator)
        self.assertIsInstance(deserialized_obj._accumulators[1],
                              SumOfSquaresAccumulator)
        self.assertEqual(deserialized_obj.compute_metrics(),
                         compound_accumulator.compute_metrics())

    def test_serialization_with_incompatible_serialized_object(self):
        mean_accumulator = MeanAccumulator(values=[15])

        serialized_obj = mean_accumulator.serialize()

        with self.assertRaises(TypeError) as context:
            SumOfSquaresAccumulator.deserialize(serialized_obj)
        self.assertEqual("The deserialized object is not of the right type.",
                         str(context.exception))

    def test_privacy_id_count(self):
        compound_accumulator1 = accumulator.CompoundAccumulator([])
        compound_accumulator2 = accumulator.CompoundAccumulator([])

        # Freshly created CompoundAccumulator has data of one privacy id.
        self.assertEqual(1, compound_accumulator1.privacy_id_count)

        # The call of add_value does not change number of privacy ids.
        compound_accumulator1.add_value(3)
        self.assertEqual(1, compound_accumulator1.privacy_id_count)

        # The count of privacy ids after addition is the sum of privacy id
        # counts because the assumption is that different CompoundAccumulator
        # have data from on-overlapping set of privacy ids.
        compound_accumulator1.add_accumulator(compound_accumulator2)
        self.assertEqual(2, compound_accumulator1.privacy_id_count)


class GenericAccumulatorTest(unittest.TestCase):

    def test_merge_accumulators(self):
        mean_accumulator1 = MeanAccumulator(values=[15])
        mean_accumulator2 = MeanAccumulator(values=[5])

        merged_accumulator = accumulator.merge(
            [mean_accumulator1, mean_accumulator2])

        self.assertEqual(merged_accumulator.compute_metrics(), 10)

        vec_params = dp_computations.AdditiveVectorNoiseParams(
            eps_per_coordinate=0,
            delta_per_coordinate=0,
            max_norm=0,
            l0_sensitivity=0,
            linf_sensitivity=0,
            norm_kind=NormKind.Linf,
            noise_kind=NoiseKind.GAUSSIAN)
        vec_sum_accumulator1 = accumulator.VectorSummationAccumulator(
            params=vec_params, values=[(15, 2)])
        vec_sum_accumulator2 = accumulator.VectorSummationAccumulator(
            params=vec_params, values=[(27, 40)])
        merged_accumulator = accumulator.merge(
            [vec_sum_accumulator1, vec_sum_accumulator2])

        with patch("pipeline_dp.dp_computations.add_noise_vector",
                   new=mock_add_noise_vector):
            self.assertEqual(tuple(merged_accumulator.compute_metrics()),
                             (42, 42))

    def test_merge_diff_type_throws_type_error(self):
        mean_accumulator1 = MeanAccumulator(values=[15])
        sum_squares_acc = SumOfSquaresAccumulator(values=[1])
        vec_params = dp_computations.AdditiveVectorNoiseParams(
            eps_per_coordinate=0,
            delta_per_coordinate=0,
            max_norm=0,
            l0_sensitivity=0,
            linf_sensitivity=0,
            norm_kind=NormKind.Linf,
            noise_kind=NoiseKind.GAUSSIAN)
        vec_sum_accumulator = accumulator.VectorSummationAccumulator(
            params=vec_params, values=[(27, 40)])

        with self.assertRaises(TypeError) as context:
            accumulator.merge([mean_accumulator1, sum_squares_acc])

        self.assertIn("The accumulator to be added is not of the same type"
                      "", str(context.exception))

        with self.assertRaises(TypeError) as context:
            accumulator.merge([vec_sum_accumulator, sum_squares_acc])

        self.assertIn("The accumulator to be added is not of the same type"
                      "", str(context.exception))

    @patch('pipeline_dp.accumulator._create_accumulator_factories')
    def test_accumulator_factory(self, mock_create_accumulator_factories):
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[agg.Metrics.MEAN],
            max_partitions_contributed=5,
            max_contributions_per_partition=3)
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=0.01)

        values = [10]
        mock_create_accumulator_factories.return_value = [
            MeanAccumulatorFactory()
        ]

        accumulator_factory = accumulator.CompoundAccumulatorFactory(
            aggregate_params, budget_accountant)
        created_accumulator = accumulator_factory.create(values)

        self.assertTrue(
            isinstance(created_accumulator, accumulator.CompoundAccumulator))
        self.assertEqual(created_accumulator.compute_metrics(), [10])
        mock_create_accumulator_factories.assert_called_with(
            aggregate_params, budget_accountant)

    @patch('pipeline_dp.accumulator._create_accumulator_factories')
    def test_accumulator_factory_multiple_types(
            self, mock_create_accumulator_factories):
        aggregate_params = pipeline_dp.AggregateParams(
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[agg.Metrics.MEAN, agg.Metrics.VAR],
            max_partitions_contributed=5,
            max_contributions_per_partition=3)
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1,
                                                  total_delta=0.01)
        values = [10]

        mock_create_accumulator_factories.return_value = [
            MeanAccumulatorFactory(),
            SumOfSquaresAccumulatorFactory()
        ]

        accumulator_factory = accumulator.CompoundAccumulatorFactory(
            aggregate_params, budget_accountant)
        created_accumulator = accumulator_factory.create(values)

        self.assertTrue(
            isinstance(created_accumulator, accumulator.CompoundAccumulator))
        self.assertEqual(created_accumulator.compute_metrics(), [10, 100])
        mock_create_accumulator_factories.assert_called_with(
            aggregate_params, budget_accountant)

    def test_create_accumulator_factories_with_count_params(self):
        acc_factories = accumulator._create_accumulator_factories(
            aggregation_params=pipeline_dp.AggregateParams(
                noise_kind=NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.COUNT],
                max_partitions_contributed=0,
                max_contributions_per_partition=0,
                budget_weight=1),
            budget_accountant=NaiveBudgetAccountant(total_epsilon=1,
                                                    total_delta=0.01))
        self.assertEqual(len(acc_factories), 1)
        self.assertIsInstance(acc_factories[0],
                              accumulator.CountAccumulatorFactory)

    def test_create_accumulator_params_with_sum_params(self):
        acc_params = accumulator._create_accumulator_factories(
            aggregation_params=pipeline_dp.AggregateParams(
                noise_kind=NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.SUM],
                max_partitions_contributed=4,
                max_contributions_per_partition=5,
                budget_weight=1),
            budget_accountant=NaiveBudgetAccountant(total_epsilon=1,
                                                    total_delta=0.01))
        self.assertEqual(len(acc_params), 1)
        self.assertIsInstance(acc_params[0], accumulator.SumAccumulatorFactory)
        self.assertTrue(isinstance(acc_params[0]._params,
                                   accumulator.SumParams))


class MeanAccumulator(accumulator.Accumulator):

    def __init__(self, values: typing.Iterable[float] = []):
        self.sum = sum(values)
        self.count = len(values)

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


class MeanAccumulatorFactory(accumulator.AccumulatorFactory):

    def create(self, values):
        return MeanAccumulator(values)


# Accumulator classes for testing
class SumOfSquaresAccumulator(accumulator.Accumulator):

    def __init__(self, values: typing.Iterable[float] = []):
        self.sum_squares = sum([value * value for value in values])

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


class SumOfSquaresAccumulatorFactory(accumulator.AccumulatorFactory):

    def create(self, values):
        return SumOfSquaresAccumulator(values)


class PrivacyIdCountAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1000000,
                                                  total_delta=0.9999999)
        budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.GAUSSIAN)
        budget_accountant.compute_budgets()
        no_noise = pipeline_dp.AggregateParams(
            low=0,
            high=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT])
        id_count_accumulator = accumulator.PrivacyIdCountAccumulator(
            accumulator.PrivacyIdCountParams(budget, no_noise), list(range(5)))
        self.assertEqual(id_count_accumulator.compute_metrics(), 1)

        id_count_accumulator = accumulator.PrivacyIdCountAccumulator(
            accumulator.PrivacyIdCountParams(budget, no_noise), 'a' * 50)
        self.assertEqual(id_count_accumulator.compute_metrics(), 1)
        id_count_accumulator.add_value(49)
        self.assertEqual(id_count_accumulator.compute_metrics(), 1)

        id_count_accumulator_1 = accumulator.PrivacyIdCountAccumulator(
            accumulator.PrivacyIdCountParams(budget, no_noise), list(range(50)))
        id_count_accumulator_2 = accumulator.PrivacyIdCountAccumulator(
            accumulator.PrivacyIdCountParams(budget, no_noise), 'a' * 50)
        id_count_accumulator_1.add_accumulator(id_count_accumulator_2)
        self.assertEqual(id_count_accumulator_1.compute_metrics(), 2)

    def test_with_noise(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=10,
                                                  total_delta=1e-5)
        budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.GAUSSIAN)
        budget_accountant.compute_budgets()

        params = pipeline_dp.AggregateParams(
            low=0,
            high=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=3,
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT])
        id_count_accumulator = accumulator.PrivacyIdCountAccumulator(
            accumulator.PrivacyIdCountParams(budget, params), list(range(5)))
        self.assertAlmostEqual(first=id_count_accumulator.compute_metrics(),
                               second=1,
                               delta=4)

        id_count_accumulator.add_value(50)
        self.assertAlmostEqual(first=id_count_accumulator.compute_metrics(),
                               second=1,
                               delta=4)


class CountAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=1000000,
                                                  total_delta=0.9999999)
        budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.GAUSSIAN)
        budget_accountant.compute_budgets()
        no_noise = pipeline_dp.AggregateParams(
            low=0,
            high=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT])
        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(budget, no_noise), list(range(5)))
        self.assertEqual(count_accumulator.compute_metrics(), 5)

        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(budget, no_noise), 'a' * 50)
        self.assertEqual(count_accumulator.compute_metrics(), 50)

        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(budget, no_noise), list(range(50)))
        count_accumulator.add_value(49)
        self.assertEqual(count_accumulator.compute_metrics(), 51)

        count_accumulator_1 = accumulator.CountAccumulator(
            accumulator.CountParams(budget, no_noise), list(range(50)))
        count_accumulator_2 = accumulator.CountAccumulator(
            accumulator.CountParams(budget, no_noise), 'a' * 50)
        count_accumulator_1.add_accumulator(count_accumulator_2)
        self.assertEqual(count_accumulator_1.compute_metrics(), 100)

    def test_with_noise(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=10,
                                                  total_delta=1e-5)
        budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.GAUSSIAN)
        budget_accountant.compute_budgets()

        params = pipeline_dp.AggregateParams(
            low=0,
            high=1,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.COUNT])
        count_accumulator = accumulator.CountAccumulator(
            accumulator.CountParams(budget, params), list(range(5)))
        self.assertAlmostEqual(first=count_accumulator.compute_metrics(),
                               second=5,
                               delta=4)

        count_accumulator.add_value(50)
        self.assertAlmostEqual(first=count_accumulator.compute_metrics(),
                               second=6,
                               delta=4)

        count_accumulator.add_value(list(range(49)))
        self.assertAlmostEqual(first=count_accumulator.compute_metrics(),
                               second=7,
                               delta=4)

        count_accumulator.add_value('*' * 100)
        self.assertAlmostEqual(first=count_accumulator.compute_metrics(),
                               second=8,
                               delta=4)


class SumAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=10000000,
                                                  total_delta=0.9999999)
        budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.GAUSSIAN)
        budget_accountant.compute_budgets()
        no_noise = pipeline_dp.AggregateParams(
            low=0,
            high=15,
            max_partitions_contributed=1,
            max_contributions_per_partition=1,
            noise_kind=NoiseKind.GAUSSIAN,
            metrics=[pipeline_dp.Metrics.SUM])
        sum_accumulator = accumulator.SumAccumulator(
            accumulator.SumParams(budget, no_noise), list(range(6)))

        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=15,
                               delta=0.1)

        sum_accumulator.add_value(5)
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=20,
                               delta=0.1)

        sum_accumulator_2 = accumulator.SumAccumulator(
            accumulator.SumParams(budget, no_noise), list(range(3)))

        sum_accumulator.add_accumulator(sum_accumulator_2)
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=23,
                               delta=0.1)

    def test_with_noise(self):
        budget_accountant = NaiveBudgetAccountant(total_epsilon=10,
                                                  total_delta=1e-5)
        budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.GAUSSIAN)
        budget_accountant.compute_budgets()

        params = pipeline_dp.AggregateParams(low=0,
                                             high=3,
                                             max_partitions_contributed=1,
                                             max_contributions_per_partition=1,
                                             noise_kind=NoiseKind.GAUSSIAN,
                                             metrics=[pipeline_dp.Metrics.SUM])
        sum_accumulator = accumulator.SumAccumulator(
            accumulator.SumParams(budget, params), list(range(3)))
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=6,
                               delta=8)

        sum_accumulator.add_value(100)  # Clamped to 3
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=9,
                               delta=8)

        sum_accumulator_2 = accumulator.SumAccumulator(
            accumulator.SumParams(budget, params), list(range(3)))
        sum_accumulator.add_accumulator(sum_accumulator_2)
        self.assertAlmostEqual(first=sum_accumulator.compute_metrics(),
                               second=12,
                               delta=8)


def mock_add_noise_vector(x, *args):
    return x


class VectorSummuationAccumulatorTest(unittest.TestCase):

    def test_without_noise(self):
        with patch("pipeline_dp.dp_computations.add_noise_vector",
                   new=mock_add_noise_vector):
            params = dp_computations.AdditiveVectorNoiseParams(
                eps_per_coordinate=0,
                delta_per_coordinate=0,
                max_norm=0,
                l0_sensitivity=0,
                linf_sensitivity=0,
                norm_kind="linf",
                noise_kind=NoiseKind.GAUSSIAN)
            vec_sum_accumulator = accumulator.VectorSummationAccumulator(
                params=params, values=[(1, 2), (3, 4), (5, 6)])
            self.assertEqual(tuple(vec_sum_accumulator.compute_metrics()),
                             (9, 12))

            vec_sum_accumulator.add_value((7, 8))
            self.assertTrue(
                np.all(vec_sum_accumulator.compute_metrics() == np.array(
                    [16, 20])))

            vec_sum_accumulator_2 = accumulator.VectorSummationAccumulator(
                params=params, values=[(1, 2), (1, 4), (1, 8), (1, 16)])
            vec_sum_accumulator.add_accumulator(vec_sum_accumulator_2)
            self.assertEqual(tuple(vec_sum_accumulator.compute_metrics()),
                             (20, 50))


if __name__ == '__main__':
    unittest.main()
