# Copyright 2022 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from apache_beam import pvalue
from unittest.mock import patch
import apache_beam.testing.util as beam_util
from apache_beam.testing.test_pipeline import TestPipeline

import pipeline_dp
from pipeline_dp import private_beam
from pipeline_dp import aggregate_params, budget_accounting


class SimplePrivatePTransform(private_beam.PrivatePTransform):

    def expand(self, pcol):
        return pcol | "Identity transform" >> beam.Map(lambda x: x)


class PrivateBeamTest(unittest.TestCase):

    @staticmethod
    def privacy_id_extractor(x):
        return f"pid:{x}"

    @staticmethod
    def value_per_key_within_tolerance(expected, actual, tolerance):
        return actual[0] == expected[0] and abs(actual[1] -
                                                expected[1]) <= tolerance
    
    @staticmethod
    def value_per_key_within_tolerance_dict(expected, actual, tolerance):
        return (actual[0] == expected[0] and (actual[1].keys() == expected[1].keys())
                and all([abs(actual[1][k] - expected[1][k]) <= tolerance for k in\
          actual[1]]))

    def test_make_private_transform_succeeds(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)

            # Act
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            # Assert
            self.assertIsInstance(private_collection,
                                  private_beam.PrivatePCollection)
            self.assertEqual(private_collection._budget_accountant,
                             budget_accountant)

    def test_private_collection_with_non_private_transform_throws_error(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            # Act and Assert
            with self.assertRaises(TypeError) as context:
                (private_collection | 'Non private transform on '
                 'PrivatePCollection' >> beam.Map(lambda x: x))
            self.assertIsInstance(private_collection,
                                  private_beam.PrivatePCollection)
            self.assertTrue(
                "private_transform should be of type "
                "PrivatePTransform but is " in str(context.exception))

    def test_transform_with_return_anonymized_disabled_returns_private_collection(
            self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            # Act
            transformer = "Simple transform" >> SimplePrivatePTransform(
                return_anonymized=False)
            transformed = private_collection | transformer

            # Assert
            self.assertEqual("Simple transform", transformer.label)
            self.assertIsInstance(transformed, private_beam.PrivatePCollection)

    def test_transform_with_return_anonymized_enabled_returns_pcollection(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            # Act
            transformed = private_collection | SimplePrivatePTransform(
                return_anonymized=True)

            # Assert
            self.assertIsInstance(transformed, pvalue.PCollection)

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_mean_calls_aggregate_with_params(self, mock_aggregate):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            mean_params = aggregate_params.MeanParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                min_value=1,
                max_value=5,
                budget_weight=1,
                public_partitions=[],
                partition_extractor=lambda x: f"pk:{x // 10}",
                value_extractor=lambda x: x)

            # Act
            transformer = private_beam.Mean(mean_params=mean_params)
            private_collection | transformer

            # Assert
            self.assertEqual(transformer._budget_accountant, budget_accountant)
            mock_aggregate.assert_called_once()

            args = mock_aggregate.call_args[0]

            params = pipeline_dp.AggregateParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.MEAN],
                max_partitions_contributed=mean_params.
                max_partitions_contributed,
                max_contributions_per_partition=mean_params.
                max_contributions_per_partition,
                min_value=mean_params.min_value,
                max_value=mean_params.max_value,
                public_partitions=mean_params.public_partitions)
            self.assertEqual(params, args[1])

    def test_mean_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(f"{u}", "pk1", -100.0) for u in range(30)]
            col += [(f"{u + 30}", "pk1", 100.0) for u in range(10)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            mean_params = aggregate_params.MeanParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=1,
                max_contributions_per_partition=1,
                min_value=1.55,
                max_value=2.7889,
                budget_weight=1,
                partition_extractor=lambda x: x[1],
                value_extractor=lambda x: x[2])

            # Act
            result = private_collection | private_beam.Mean(
                mean_params=mean_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pk1", 1.859)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 0.1)))

    def test_mean_with_public_partitions_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(f"{u}", "pubK1", -100.0) for u in range(30)]
            col += [(f"{u + 30}", "pubK1", 100.0) for u in range(10)]
            col += [(f"{u + 40}", "privK1", 100.0) for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            mean_params = aggregate_params.MeanParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=1,
                max_contributions_per_partition=1,
                min_value=1.55,
                max_value=2.7889,
                budget_weight=1,
                partition_extractor=lambda x: x[1],
                value_extractor=lambda x: x[2],
                public_partitions=["pubK1", "pubK2"])

            # Act
            result = private_collection | private_beam.Mean(
                mean_params=mean_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pubK1", 1.859), ("pubK2", 2.1695)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 0.1)))

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_sum_calls_aggregate_with_params(self, mock_aggregate):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                float(i) for i in range(1, 7))
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            sum_params = aggregate_params.SumParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                min_value=1,
                max_value=5,
                budget_weight=1,
                public_partitions=[],
                partition_extractor=lambda x: f"pk:{x // 10}",
                value_extractor=lambda x: x)

            # Act
            transformer = private_beam.Sum(sum_params=sum_params)
            private_collection | transformer

            # Assert
            self.assertEqual(transformer._budget_accountant, budget_accountant)
            mock_aggregate.assert_called_once()

            args = mock_aggregate.call_args[0]

            params = pipeline_dp.AggregateParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.SUM],
                max_partitions_contributed=sum_params.
                max_partitions_contributed,
                max_contributions_per_partition=sum_params.
                max_contributions_per_partition,
                min_value=sum_params.min_value,
                max_value=sum_params.max_value,
                public_partitions=sum_params.public_partitions)
            self.assertEqual(params, args[1])

    def test_sum_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(f"{u}", "pk1", 100.0) for u in range(30)]
            col += [(f"{u + 30}", "pk1", -100.0) for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            sum_params = aggregate_params.SumParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                min_value=1.55,
                max_value=2.7889,
                budget_weight=1,
                partition_extractor=lambda x: x[1],
                value_extractor=lambda x: x[2])

            # Act
            result = private_collection | private_beam.Sum(
                sum_params=sum_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pk1", 130.167)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 10.0)))

    def test_sum_with_public_partitions_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(f"{u}", "pubK1", 100.0) for u in range(30)]
            col += [(f"{u + 30}", "pubK1", -100.0) for u in range(30)]
            col += [(f"{u + 60}", "privK1", 100.0) for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            sum_params = aggregate_params.SumParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                min_value=1.55,
                max_value=2.7889,
                budget_weight=1,
                partition_extractor=lambda x: x[1],
                value_extractor=lambda x: x[2],
                public_partitions=["pubK1", "pubK2"])

            # Act
            result = private_collection | private_beam.Sum(
                sum_params=sum_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pubK1", 130.167), ("pubK2", 0.0)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 10.0)))

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_count_calls_aggregate_with_params(self, mock_aggregate):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            count_params = aggregate_params.CountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                budget_weight=1,
                partition_extractor=lambda x: f"pk:{x // 10}")

            # Act
            transformer = private_beam.Count(count_params=count_params)
            private_collection | transformer

            # Assert
            self.assertEqual(transformer._budget_accountant, budget_accountant)
            mock_aggregate.assert_called_once()

            args = mock_aggregate.call_args[0]

            params = pipeline_dp.AggregateParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.COUNT],
                max_partitions_contributed=count_params.
                max_partitions_contributed,
                max_contributions_per_partition=count_params.
                max_contributions_per_partition,
                public_partitions=count_params.public_partitions)
            self.assertEqual(args[1], params)

    def test_count_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(u, "pk1") for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            count_params = aggregate_params.CountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                budget_weight=1,
                partition_extractor=lambda x: x[1])

            # Act
            result = private_collection | private_beam.Count(
                count_params=count_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pk1", 30.0)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 5.0)))

    def test_count_with_public_partitions_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(u, "pubK1") for u in range(30)]
            col += [(u, "privK1") for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            count_params = aggregate_params.CountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                budget_weight=1,
                partition_extractor=lambda x: x[1],
                public_partitions=["pubK1", "pubK2"])

            # Act
            result = private_collection | private_beam.Count(
                count_params=count_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pubK1", 30.0), ("pubK2", 0.0)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 5.0)))

    @patch('pipeline_dp.dp_engine.DPEngine.aggregate')
    def test_privacy_id_count_calls_aggregate_with_params(self, mock_aggregate):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            privacy_id_count_params = aggregate_params.PrivacyIdCountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                budget_weight=1,
                partition_extractor=lambda x: f"pk:{x // 10}")

            # Act
            transformer = private_beam.PrivacyIdCount(
                privacy_id_count_params=privacy_id_count_params)
            private_collection | transformer

            # Assert
            self.assertEqual(transformer._budget_accountant, budget_accountant)
            mock_aggregate.assert_called_once()

            args = mock_aggregate.call_args[0]

            params = pipeline_dp.AggregateParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                metrics=[pipeline_dp.Metrics.PRIVACY_ID_COUNT],
                max_partitions_contributed=privacy_id_count_params.
                max_partitions_contributed,
                max_contributions_per_partition=1,
                public_partitions=privacy_id_count_params.public_partitions)
            self.assertEqual(args[1], params)

    def test_privacy_id_count_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(u, "pk1") for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            privacy_id_count_params = aggregate_params.PrivacyIdCountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                budget_weight=1,
                partition_extractor=lambda x: x[1])

            # Act
            result = private_collection | private_beam.PrivacyIdCount(
                privacy_id_count_params=privacy_id_count_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pk1", 30)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 5)))

    def test_privacy_id_count_with_public_partitions_returns_sensible_result(
            self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(u, "pubK1") for u in range(30)]
            col += [(u, "privK1") for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            privacy_id_count_params = aggregate_params.PrivacyIdCountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                budget_weight=1,
                partition_extractor=lambda x: x[1],
                public_partitions=["pubK1", "pubK2"])

            # Act
            result = private_collection | private_beam.PrivacyIdCount(
                privacy_id_count_params=privacy_id_count_params)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pubK1", 30.0), ("pubK2", 0.0)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 5.0)))

    def test_map_returns_correct_results_and_accountant(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol_input = [(1, 2), (2, 3), (3, 4), (4, 5)]
            pcol = pipeline | 'Create produce' >> beam.Create(pcol_input)
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            # Act
            transformed = private_collection | private_beam.Map(
                fn=lambda x: x[1]**2)

            # Assert
            self.assertIsInstance(transformed, private_beam.PrivatePCollection)
            beam_util.assert_that(
                transformed._pcol,
                beam_util.equal_to(
                    map(
                        lambda x:
                        (PrivateBeamTest.privacy_id_extractor(x), x[1]**2),
                        pcol_input)))
            self.assertEqual(transformed._budget_accountant, budget_accountant)

    def test_flatmap_returns_correct_results_and_accountant(self):

        def flat_map_fn(x):
            return [(x[0], x[1] + i) for i in range(2)]

        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol_input = [(1, 2), (2, 3), (3, 4)]
            pcol = pipeline | 'Create produce' >> beam.Create(pcol_input)
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            # Act
            transformed = private_collection | private_beam.FlatMap(flat_map_fn)

            # Assert
            self.assertIsInstance(transformed, private_beam.PrivatePCollection)
            beam_util.assert_that(
                transformed._pcol,
                beam_util.equal_to([('pid:(1, 2)', (1, 2)),
                                    ('pid:(1, 2)', (1, 3)),
                                    ('pid:(2, 3)', (2, 3)),
                                    ('pid:(2, 3)', (2, 4)),
                                    ('pid:(3, 4)', (3, 4)),
                                    ('pid:(3, 4)', (3, 5))]))
            self.assertEqual(transformed._budget_accountant, budget_accountant)

    @patch('pipeline_dp.dp_engine.DPEngine.select_partitions')
    def test_select_partitions_calls_select_partitions_with_params(
            self, mock_select_partitions):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # Arrange
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=PrivateBeamTest.privacy_id_extractor))

            select_partitions_params = \
                aggregate_params.SelectPartitionsParams(
                    max_partitions_contributed=2,
                    budget_weight=0.5,)
            partition_extractor = lambda x: f"pk:{x // 10}"

            # Act
            transformer = private_beam.SelectPartitions(
                select_partitions_params=select_partitions_params,
                partition_extractor=partition_extractor,
                label="Test select partitions")
            private_collection | transformer

            # Assert
            self.assertEqual(transformer._budget_accountant, budget_accountant)
            mock_select_partitions.assert_called_once()

            args = mock_select_partitions.call_args[0]
            self.assertEqual(args[1], select_partitions_params)

    def test_select_private_partitions_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(u, "pk1") for u in range(50)]
            col += [(50 + u, "pk2") for u in range(50)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            select_partitions_params = \
                aggregate_params.SelectPartitionsParams(
                    max_partitions_contributed=2,
                    budget_weight=0.9)
            partition_extractor = lambda x: x[1]

            # Act
            result = private_collection | private_beam.SelectPartitions(
                select_partitions_params=select_partitions_params,
                partition_extractor=partition_extractor,
                label="Test select partitions")
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(result, beam_util.equal_to(["pk1", "pk2"]))

    def test_combine_per_returns_sensible_result(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(f"{u}", "pk1", 100.0) for u in range(30)]
            col += [(f"{u + 30}", "pk1", -100.0) for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
                pcol | 'Create private collection' >> private_beam.MakePrivate(
                    budget_accountant=budget_accountant,
                    privacy_id_extractor=lambda x: x[0]))

            private_collection = private_collection | private_beam.Map(
                lambda x: (x[1], x[2]))

            # Act
            result = private_collection | private_beam.CombinePerKey(
                SumCombineFn(),
                private_beam.CombinePerKeyParams(
                    max_partitions_contributed=2,
                    max_contributions_per_partition=1))
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pk1", 0.0)],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance(e, a, 10.0)))

    def test_multiple_aggregates(self):
        with TestPipeline() as pipeline:
            # Arrange
            col = [(u, "pk1") for u in range(30)]
            pcol = pipeline | 'Create produce' >> beam.Create(col)
            # Use very high epsilon and delta to minimize noise and test
            # flakiness.
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=800, total_delta=0.999)
            private_collection = (
              pcol | 'Create private collection' >> private_beam.MakePrivate(
                budget_accountant=budget_accountant,
                privacy_id_extractor=lambda x: x[0]))

            privacy_id_count_params = aggregate_params.PrivacyIdCountParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                budget_weight=1,
                partition_extractor=lambda x: x[1])

            # Act
            result = private_collection | private_beam.Aggregate().aggregate_value(
                privacy_id_count_params,
                col_name='privacy_id_count',
                agg_type=pipeline_dp.Metrics.PRIVACY_ID_COUNT)
            budget_accountant.compute_budgets()

            # Assert
            # This is a health check to validate that the result is sensible.
            # Hence, we use a very large tolerance to reduce test flakiness.
            beam_util.assert_that(
                result,
                beam_util.equal_to([("pk1", {'privacy_id_count':30})],
                                   equals_fn=lambda e, a: PrivateBeamTest.
                                   value_per_key_within_tolerance_dict(e, a, 5)))

class SumCombineFn(private_beam.PrivateCombineFn):
    """Test-only, not private combine_fn."""

    def create_accumulator(self):
        return 0

    def add_input_for_private_output(self, accumulator, input):
        return accumulator + input

    def merge_accumulators(self, accumulators):
        return sum(accumulators)

    def extract_private_output(self, accumulator, budget):
        return accumulator

    def request_budget(self, budget_accountant):
        return None


if __name__ == '__main__':
    unittest.main()
