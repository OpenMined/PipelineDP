import unittest
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from apache_beam import pvalue
from unittest.mock import patch
import apache_beam.testing.util as beam_util

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

    def test_MakePrivate_transform_succeeds(self):
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
            transformed = private_collection | SimplePrivatePTransform(
                return_anonymized=False)

            # Assert
            self.assertIsInstance(transformed, private_beam.PrivatePCollection)

    def test_transform_with_return_anonymized_enabled_returns_PCollection(self):
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
    def test_sum(self, mock_aggregate):
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

            sum_params = aggregate_params.SumParams(
                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                low=1,
                high=5,
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
                low=sum_params.low,
                high=sum_params.high,
                public_partitions=sum_params.public_partitions)
            self.assertEqual(args[1], params)

    def test_Map(self):
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

    def test_FlatMap(self):

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


if __name__ == '__main__':
    unittest.main()
