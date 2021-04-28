""" The example of using DPEngine for performing DP aggregation.

Warning: DP aggregations has not been implemented yet, this example is only for
demonstration of the API and development purposes.

In order to run an example:

1.Install Python and run in command line pip install numpy apache-beam absl-py
2.Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.
3.The dataset itself is pretty big, for speed-up the run it's better to use a
part of it. You can generate a part of it by running in bash:

   head -10000 combined_data_1.txt > data.txt

   or by other way to get a subset of lines from the dataset.

4. Run python movie_view_ratings.py --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from dataclasses import dataclass
import pipeline_dp

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', '', 'The file with the movie view  data')
flags.DEFINE_string('output_file', None, 'Output file')


@dataclass
class MovieView:
  user_id: int
  movie_id: int
  rating: int


def parse_line(line, movie_id):
  # 'line' has format "user_id,rating,date"
  split_parts = line.split(',')
  user_id = int(split_parts[0])
  rating = int(split_parts[1])
  return MovieView(user_id, movie_id, rating)


class ParseFile(beam.DoFn):

  def __init__(self):
    self.movie_id = -1

  def process(self, line):
    if line[-1] == ':':
      # 'line' has a format "movie_id:'
      self.movie_id = int(line[:-1])
      return
    # 'line' has a format "user_id,rating,date"
    yield parse_line(line, self.movie_id)


def calc_dp_rating_metrics(pipeline):
  """Computes dp metrics."""

  movie_views = pipeline | beam.io.ReadFromText(FLAGS.input_file) | beam.ParDo(
      ParseFile())

  # Set the total privacy budget.
  budget_accountant = pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-6)

  # Specify a pipeline framework to use.
  ops = pipeline_dp.BeamOperations()

  # Create a DPEngine instance.
  dp_engine = pipeline_dp.DPEngine(budget_accountant, ops)

  # Specify which DP aggregated metrics to compute.
  params = pipeline_dp.AggregateParams(
      metrics=[
          pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT,
          pipeline_dp.Metrics.MEAN
      ],
      max_partitions_contributed=2,
      max_contributions_per_partition=1,
      low=1,
      high=5,
  )

  # Specify how to extract is privacy_id, partition_key and value from an element of movie view collection.
  data_extractors = pipeline_dp.DataExtractors(
      partition_extractor=lambda mv: mv.movie_id,
      privacy_id_extractor=lambda mv: mv.user_id,
      value_extractor=lambda mv: mv.rating)

  # Run aggregation.
  dp_result = dp_engine.aggregate(movie_views, params, data_extractors)

  return dp_result


def compute_on_beam():
  runner = fn_api_runner.FnApiRunner()  # local runner
  with beam.Pipeline(runner=runner) as pipeline:
    dp_result = calc_dp_rating_metrics(pipeline)
    dp_result | beam.io.WriteToText(FLAGS.output_file)


def main(unused_argv):
  compute_on_beam()
  return 0


if __name__ == '__main__':
  app.run(main)
