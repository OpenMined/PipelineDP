import unittest
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner

from pipeline_dp import private_beam
from pipeline_dp import aggregate_params, budget_accounting


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


if __name__ == '__main__':
    unittest.main()
