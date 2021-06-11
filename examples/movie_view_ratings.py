""" The example of using DPEngine for performing DP aggregation.

Warning: DP aggregations has not been implemented yet, this example is only for
demonstration of the API and development purposes.

In order to run an example:

1.Install Python and run in command line pip install numpy apache-beam pyspark absl-py
2.Download Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.
3.The dataset itself is pretty big, for speed-up the run it's better to use a
part of it. You can generate a part of it by running in bash:

   head -10000 combined_data_1.txt > data.txt

   or by other way to get a subset of lines from the dataset.

4. Run python movie_view_ratings.py --framework=<framework> --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
import pyspark
from dataclasses import dataclass
import pipeline_dp

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', '', 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_enum('framework', None, ['beam', 'spark'], 'Pipeline framework to use.')
flags.DEFINE_list('public_partitions', None, 'List of comma-separated public partition keys')

@dataclass
class MovieView:
    user_id: int
    movie_id: int
    rating: int


def calc_dp_rating_metrics(movie_views, ops):
    """Computes dp metrics."""

    # Set the total privacy budget.
    budget_accountant = pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-6)

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


def compute_on_beam():
    runner = fn_api_runner.FnApiRunner()  # local runner
    public_partitions=[int(partition) for partition in FLAGS.public_partitions] if FLAGS.public_partitions is not None else None
    with beam.Pipeline(runner=runner) as pipeline:
        movie_views = pipeline | beam.io.ReadFromText(FLAGS.input_file) | beam.ParDo(
            ParseFile())
        pipeline_operations = pipeline_dp.BeamOperations()
        dp_result = calc_dp_rating_metrics(movie_views, pipeline_operations, public_partitions)
        dp_result | beam.io.WriteToText(FLAGS.output_file)


def parse_partition(iterator):
    movie_id = None
    for line in iterator:
        if line[-1] == ':':
            # 'line' has a format "movie_id:'
            movie_id = int(line[:-1])
        else:
            # 'line' has a format "user_id,rating,date"
            yield parse_line(line, movie_id)


def compute_on_spark():
    master = "local[1]"  # run Spark locally with one worker thread to load the input file into 1 partition
    conf = pyspark.SparkConf().setMaster(master)
    sc = pyspark.SparkContext(conf=conf)
    movie_views = sc.textFile(FLAGS.input_file) \
        .mapPartitions(parse_partition)
    pipeline_operations = pipeline_dp.SparkRDDOperations()
    dp_result = calc_dp_rating_metrics(movie_views, pipeline_operations)
    dp_result.saveAsTextFile(FLAGS.output_file)


def compute_on_local():
    raise NotImplementedError


def main(unused_argv):
    if FLAGS.framework == 'beam':
        compute_on_beam()
    elif FLAGS.framework == 'spark':
        compute_on_spark()
    else:
        compute_on_local()
    return 0


if __name__ == '__main__':
    app.run(main)
