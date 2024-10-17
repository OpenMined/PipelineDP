import unittest

import pipeline_dp
from pipeline_dp.dataset_histograms import histograms as hist
from pipeline_dp import private_contribution_bounds


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
                                                            upper=2,
                                                            count=100,
                                                            sum=100,
                                                            min=1,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            upper=6,
                                                            count=10,
                                                            sum=10,
                                                            min=2,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            upper=100,
                                                            count=20,
                                                            sum=20,
                                                            min=6,
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

    def test_score_gaussian_noise_valid_values_calculates_score_correctly(self):
        params = construct_params(
            aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            aggregation_eps=0.9,
            aggregation_delta=0.001,
            max_partitions_contributed_upper_bound=100)
        number_of_partitions = 200
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            upper=2,
                                                            count=100,
                                                            sum=100,
                                                            min=1,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            upper=6,
                                                            count=10,
                                                            sum=10,
                                                            min=2,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            upper=100,
                                                            count=20,
                                                            sum=20,
                                                            min=6,
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

        # sigma = gaussian_std(eps=0.9, delta=0.001) = 2.8127255538420286
        # -0.5 * 200 * sqrt(1) * sigma - 0.5 * ((2 - 1) * 10 + (6 - 1) * 20)
        self.assertAlmostEqual(-336, score_1, delta=10)
        # -0.5 * 200 * sqrt(2) * sigma - 0.5 * ((6 - 2) * 20)
        self.assertAlmostEqual(-437, score_2, delta=10)
        # -0.5 * 200 * sqrt(5) * sigma - 0.5 * ((6 - 5) * 20)
        self.assertAlmostEqual(-638, score_5, delta=10)
        # -0.5 * 200 * sqrt(6) * sigma - 0.5 * 0
        self.assertAlmostEqual(-688, score_6, delta=10)
        # -0.5 * 200 * sqrt(60) * sigma - 0.5 * 0
        self.assertAlmostEqual(-2178, score_60, delta=10)
        # -0.5 * 200 * sqrt(61) * sigma - 0.5 * 0
        self.assertAlmostEqual(-2196, score_61, delta=10)
        # -0.5 * 200 * sqrt(101) * sigma - 0.5 * 0
        self.assertAlmostEqual(-2826, score_101, delta=10)
        # -0.5 * 200 * sqrt(200) * sigma - 0.5 * 0
        self.assertAlmostEqual(-3977, score_200, delta=10)


class PrivateL0CalculatorTest(unittest.TestCase):

    def test_calculate_returns_one_of_the_lower_bounds(self):
        # Arrange
        params = construct_params(
            aggregation_eps=0.9,
            aggregation_delta=1e-10,
            calculation_eps=0.1,
            aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed_upper_bound=100)
        partitions = [i + 1 for i in range(200)]
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            upper=2,
                                                            count=100,
                                                            sum=100,
                                                            min=1,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            upper=6,
                                                            count=10,
                                                            sum=10,
                                                            min=2,
                                                            max=5),
                                          hist.FrequencyBin(lower=6,
                                                            upper=100,
                                                            count=20,
                                                            sum=20,
                                                            min=6,
                                                            max=60)
                                      ])
        histograms = [
            hist.DatasetHistograms(l0_histogram,
                                   l1_contributions_histogram=None,
                                   linf_contributions_histogram=None,
                                   linf_sum_contributions_histogram=None,
                                   count_per_partition_histogram=None,
                                   count_privacy_id_per_partition=None,
                                   sum_per_partition_histogram=None)
        ]
        backend = pipeline_dp.LocalBackend()
        calculator = private_contribution_bounds.PrivateL0Calculator(
            params, partitions, histograms, backend)

        # Act
        l0_bound = list(calculator.calculate())[0]

        # Assert
        self.assertIn(l0_bound, list(range(1, 101)))

    def test_calculate_one_bound_has_much_higher_score_returns_it(self):
        # Arrange
        params = construct_params(
            aggregation_eps=0.9,
            aggregation_delta=0,
            calculation_eps=0.1,
            aggregation_noise_kind=pipeline_dp.NoiseKind.LAPLACE,
            max_partitions_contributed_upper_bound=2)
        partitions = list(range(1, 201))
        l0_histogram = hist.Histogram(name=hist.HistogramType.L0_CONTRIBUTIONS,
                                      bins=[
                                          hist.FrequencyBin(lower=1,
                                                            upper=2,
                                                            count=1,
                                                            sum=1,
                                                            min=1,
                                                            max=1),
                                          hist.FrequencyBin(lower=2,
                                                            upper=3,
                                                            count=10000,
                                                            sum=10000,
                                                            min=2,
                                                            max=2)
                                      ])
        histograms = [
            hist.DatasetHistograms(l0_histogram,
                                   l1_contributions_histogram=None,
                                   linf_contributions_histogram=None,
                                   linf_sum_contributions_histogram=None,
                                   count_per_partition_histogram=None,
                                   count_privacy_id_per_partition=None,
                                   sum_per_partition_histogram=None)
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

    def test_generate_possible_contribution_bounds(self):
        # Even though it is private function, we test it explicitly because
        # testing it through public API requires a complicated test setup.
        upper_bound = 999999

        bounds = private_contribution_bounds.generate_possible_contribution_bounds(
            upper_bound)

        expected = list(range(1, 1000, 1)) + list(range(
            1000, 10000, 10)) + list(range(10000, 100000, 100)) + list(
                range(100000, 1000000, 1000))
        self.assertEqual(bounds, expected)


if __name__ == '__main__':
    unittest.main()
