import unittest
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from apache_beam.pvalue import PCollection

from pipeline_dp import private_beam
from pipeline_dp import aggregate_params, budget_accounting


class SimplePrivateTransform(private_beam.PrivateTransform):

    def expand(self, pcol):
        return pcol | "Identity transform" >> beam.Map(lambda x: x)


class MyTestCase(unittest.TestCase):

    def test_MakePrivate_transform_succeeds(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # assert
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            privacy_id_extractor = lambda x: "pid:" + x

            # act
            private_collection = pcol \
                  | 'Create private collection' >> private_beam.MakePrivate(
              budget_accountant=budget_accountant,
              privacy_id_extractor=privacy_id_extractor)

            # assert
            self.assertIsInstance(private_collection,
                                  private_beam.PrivateCollection)
            self.assertEqual(private_collection.budget_accountant,
                             budget_accountant)
            self.assertEqual(private_collection.privacy_id_extractor,
                             privacy_id_extractor)

    def test_private_collection_with_non_private_transform_throws_error(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # assert
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            privacy_id_extractor = lambda x: "pid:" + x
            private_collection = pcol \
                  | 'Create private collection' >> private_beam.MakePrivate(
              budget_accountant=budget_accountant,
              privacy_id_extractor=privacy_id_extractor)

            # act and assert
            with self.assertRaises(TypeError) as context:
                private_collection | 'Non private transform on ' \
                                     'privateCollection' >> beam.Map(lambda x: x)

                self.assertIsInstance(private_collection,
                                      private_beam.PrivateCollection)
                self.assertTrue("pcol should of type PrivateCollection but is"
                                in str(context.exception))

    def test_private_collection_with_make_private_enabled_returns_private_collection(
            self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # assert
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            privacy_id_extractor = lambda x: "pid:" + x
            private_collection = pcol \
                  | 'Create private collection' >> private_beam.MakePrivate(
              budget_accountant=budget_accountant,
              privacy_id_extractor=privacy_id_extractor)

            # Act
            private_transformer = SimplePrivateTransform()
            private_transformer.set_return_private()

            transformed = private_collection | private_transformer

            # Assert
            self.assertIsInstance(transformed, private_beam.PrivateCollection)

    def test_private_collection_with_make_private_disabled_returns_PCollection(
            self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # assert
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            budget_accountant = budget_accounting.NaiveBudgetAccountant(
                total_epsilon=1, total_delta=0.01)
            privacy_id_extractor = lambda x: "pid:" + x
            private_collection = pcol \
                  | 'Create private collection' >> private_beam.MakePrivate(
              budget_accountant=budget_accountant,
              privacy_id_extractor=privacy_id_extractor)

            # Act
            transformed = private_collection | SimplePrivateTransform()

            # Assert
            self.assertIsInstance(transformed, PCollection)

    def test_Sum_transform_with_PCollection_throws_Exception(self):
        runner = fn_api_runner.FnApiRunner()
        with beam.Pipeline(runner=runner) as pipeline:
            # assert
            pcol = pipeline | 'Create produce' >> beam.Create(
                [1, 2, 3, 4, 5, 6])
            sum_params = aggregate_params.SumParams(
                max_partitions_contributed=2,
                max_contributions_per_partition=3,
                low=1,
                high=5,
                budget_weight=1,
                public_partitions=None)

            # act and assert
            with self.assertRaises(TypeError) as context:
                pcol | 'Non private transform on ' \
                                     'privateCollection' >> private_beam.Sum(sum_params, lambda x: (x % 10))

                self.assertIsInstance(pcol, PCollection)
                self.assertTrue("pcol should of type PrivateCollection but is"
                                in str(context.exception))


if __name__ == '__main__':
    unittest.main()
