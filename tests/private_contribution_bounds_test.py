import unittest

import pipeline_dp
from pipeline_dp import NaiveBudgetAccountant, PipelineBackend, \
    CalculatePrivateContributionBoundsParams, PrivateContributionBounds
from pipeline_dp import histograms as hist
from pipeline_dp import private_contribution_bounds
from pipeline_dp.report_generator import ReportGenerator


def construct_params(
    aggregation_noise_kind=pipeline_dp.NoiseKind.LAPLACE,
    aggregation_eps=0.9,
    aggregation_delta=0.0,
    calculation_eps=0.1,
    max_partitions_contributed_upper_bound=3
) -> pipeline_dp.CalculatePrivateContributionBoundsParams:
    return pipeline_dp.CalculatePrivateContributionBoundsParams(
        aggregation_noise_kind, aggregation_eps, aggregation_delta,
        calculation_eps, max_partitions_contributed_upper_bound)


class L0ScoringFunctionTest(unittest.TestCase):

    def test_is_monotonic_is_always_true(self):
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params=None, number_of_partitions=0, l0_histogram=None)

        self.assertEqual(True, l0_scoring_function.is_monotonic)

    def test_global_sensitivity_equals_upper_bound_when_it_is_precise(self):
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params=construct_params(max_partitions_contributed_upper_bound=3),
            number_of_partitions=100,
            l0_histogram=None)

        self.assertEqual(3, l0_scoring_function.global_sensitivity)

    def test_global_sensitivity_equals_number_of_partitions_when_upper_bound_is_bigger_than_number_of_partitions(
            self):
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params=construct_params(max_partitions_contributed_upper_bound=100),
            number_of_partitions=10,
            l0_histogram=None)

        self.assertEqual(10, l0_scoring_function.global_sensitivity)

    def test_score_laplace_noise_valid_values_calculates_score_correctly(self):
        params = construct_params(
            aggregation_noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            aggregation_eps=0.9,
            max_partitions_contributed_upper_bound=100)
        number_of_partitions = 200
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            count=100,
                                                            sum=100,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            count=10,
                                                            sum=10,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            count=20,
                                                            sum=20,
                                                            max=60)
                                      ])
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params, number_of_partitions, l0_histogram)

        score_1 = l0_scoring_function.score(1)
        score_2 = l0_scoring_function.score(2)
        score_5 = l0_scoring_function.score(5)
        score_6 = l0_scoring_function.score(6)
        score_60 = l0_scoring_function.score(60)
        score_61 = l0_scoring_function.score(61)
        score_101 = l0_scoring_function.score(101)
        score_200 = l0_scoring_function.score(200)

        # -0.5 * 200 * 1 / 0.9 * sqrt(2) - 0.5 * ((2 - 1) * 10 + (6 - 1) * 20)
        self.assertAlmostEqual(-212.135, score_1, places=3)
        # -0.5 * 200 * 2 / 0.9 * sqrt(2) - 0.5 * ((6 - 2) * 20)
        self.assertAlmostEqual(-354.270, score_2, places=3)
        # -0.5 * 200 * 5 / 0.9 * sqrt(2) - 0.5 * ((6 - 5) * 20)
        self.assertAlmostEqual(-795.674, score_5, places=3)
        # -0.5 * 200 * 6 / 0.9 * sqrt(2) - 0.5 * 0
        self.assertAlmostEqual(-942.809, score_6, places=3)
        # -0.5 * 200 * 60 / 0.9 * sqrt(2) - 0.5 * 0
        self.assertAlmostEqual(-9428.09, score_60, places=2)
        # -0.5 * 200 * 61 / 0.9 * sqrt(2) - 0.5 * 0
        self.assertAlmostEqual(-9585.23, score_61, places=2)
        # -0.5 * 200 * 101 / 0.9 * sqrt(2) - 0.5 * 0
        self.assertAlmostEqual(-15870.6, score_101, places=1)
        # -0.5 * 200 * 200 / 0.9 * sqrt(2) - 0.5 * 0
        self.assertAlmostEqual(-31427.0, score_200, places=1)

    def test_score_laplace_noise_invalid_values_throws_exception(self):
        params = construct_params(
            aggregation_noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            aggregation_eps=0.9,
            max_partitions_contributed_upper_bound=100)
        number_of_partitions = 200
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            count=100,
                                                            sum=100,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            count=10,
                                                            sum=10,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            count=20,
                                                            sum=20,
                                                            max=60)
                                      ])
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params, number_of_partitions, l0_histogram)

        with self.assertRaises(RuntimeError):
            l0_scoring_function.score(-1)

        with self.assertRaises(RuntimeError):
            l0_scoring_function.score(0)

    def test_score_gaussian_noise_valid_value_returns_value_of_a_correct_order_of_magnitude(
            self):
        # We don't check for exact value because PyDP uses different algorithm
        # to calculate std of a Gaussian noise. Therefore, we just check
        # the order of magnitude of the returned value and also check
        # that for Gaussian noise l0 scoring function works in general.
        params = construct_params(
            aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            aggregation_eps=0.9,
            aggregation_delta=0.001,
            max_partitions_contributed_upper_bound=100)
        number_of_partitions = 200
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            count=100,
                                                            sum=100,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            count=10,
                                                            sum=10,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            count=20,
                                                            sum=20,
                                                            max=60)
                                      ])
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params, number_of_partitions, l0_histogram)

        score = l0_scoring_function.score(1)

        self.assertLess(-1000, score)
        self.assertGreater(0, score)

    def test_score_gaussian_noise_invalid_values_throws_exception(self):
        params = construct_params(
            aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            aggregation_eps=0.9,
            aggregation_delta=0.001,
            max_partitions_contributed_upper_bound=100)
        number_of_partitions = 200
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            count=100,
                                                            sum=100,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            count=10,
                                                            sum=10,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            count=20,
                                                            sum=20,
                                                            max=60)
                                      ])
        l0_scoring_function = private_contribution_bounds.L0ScoringFunction(
            params, number_of_partitions, l0_histogram)

        with self.assertRaises(RuntimeError):
            l0_scoring_function.score(-1)

        with self.assertRaises(RuntimeError):
            l0_scoring_function.score(0)


class PrivateL0CalculatorTest(unittest.TestCase):

    def test_calculate_returns_one_of_the_lower_bounds(self):
        # Arrange
        params = construct_params(
            aggregation_eps=0.9,
            aggregation_delta=1e-10,
            calculation_eps=0.1,
            aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed_upper_bound=2)
        partitions = [i + 1 for i in range(200)]
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            count=1000,
                                                            sum=1000,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            count=1,
                                                            sum=1,
                                                            max=5)
                                      ])
        histograms = [
            hist.DatasetHistograms(l0_histogram,
                                   linf_contributions_histogram=None,
                                   count_per_partition_histogram=None,
                                   count_privacy_id_per_partition=None)
        ]
        backend = pipeline_dp.LocalBackend()
        calculator = private_contribution_bounds.PrivateL0Calculator(
            params, partitions, histograms, backend)

        # Act
        l0_bound = list(calculator.calculate())[0]

        # Assert
        self.assertIn(l0_bound, [1, 2, 6])

    def test_calculate_one_bound_has_much_higher_score_returns_it(self):
        # Arrange
        params = construct_params(
            aggregation_eps=0.9,
            aggregation_delta=0,
            calculation_eps=0.1,
            aggregation_noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            max_partitions_contributed_upper_bound=2)
        partitions = [i + 1 for i in range(200)]
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            count=1,
                                                            sum=1,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            count=10000,
                                                            sum=10000,
                                                            max=2)
                                      ])
        histograms = [
            hist.DatasetHistograms(l0_histogram,
                                   linf_contributions_histogram=None,
                                   count_per_partition_histogram=None,
                                   count_privacy_id_per_partition=None)
        ]
        backend = pipeline_dp.LocalBackend()
        calculator = private_contribution_bounds.PrivateL0Calculator(
            params, partitions, histograms, backend)

        # Act
        l0_bound = list(calculator.calculate())[0]

        # Assert
        # score(1) = -0.5 * 200 * 1 / 0.9 * sqrt(2) - 0.5 * ((2 - 1) * 10000) =
        # -5157
        # score(2) = -0.5 * 200 * 2 / 0.9 * sqrt(2) - 0.5 * 0 = -314
        # probability = e^(0.1 * score / 2) / total
        # e^(0.1 * (-5157) / 2) = 1.04e-112
        # e^(0.1 * (-314) / 2) = 1.52e-7
        # probability of 2 = 1, i.e. 2 has to be returned
        self.assertEqual(l0_bound, 2)


if __name__ == '__main__':
    unittest.main()
