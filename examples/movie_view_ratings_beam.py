""" Demo of PipelineDP with Apache Beam.
"""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
import pipeline_dp
from pipeline_dp import private_beam
from pipeline_dp import SumParams
from pipeline_dp.private_beam import MakePrivate
from examples.example_utils import ParseFile

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')


def main(unused_argv):
    # Setup Beam

    # Here, we use a local Beam runner.
    # For a truly distributed calculation, connect to a Spark cluster (e.g.
    # running on some cloud provider).
    runner = fn_api_runner.FnApiRunner()  # Local Beam runner
    with beam.Pipeline(runner=runner) as pipeline:

        # Define the privacy budget available for our computation.
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-6)

        # Load and parse input data
        movie_views_pcol = pipeline | \
                           beam.io.ReadFromText(FLAGS.input_file) | \
                           beam.ParDo(ParseFile())

        # Wrap Beam's PCollection into it's private version
        private_movie_views = (movie_views_pcol |
                               'Create private collection' >> MakePrivate(
                                   budget_accountant=budget_accountant,
                                   privacy_id_extractor=lambda mv: mv.user_id))

        # Calculate the private sum
        dp_result = private_movie_views | private_beam.Sum(
            SumParams(max_partitions_contributed=2,
                      max_contributions_per_partition=2,
                      low=1,
                      high=5,
                      partition_extractor=lambda mv: mv.movie_id,
                      value_extractor=lambda mv: mv.rating))
        budget_accountant.compute_budgets()

        # Save the results
        dp_result | beam.io.WriteToText(FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
