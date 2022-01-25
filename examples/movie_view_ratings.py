""" The example of using DPEngine for performing DP aggregation.

This is a quite elaborate example demonstrating many features. For a simpler
example of how to use PipelineDP with spark, check
movie_view_ratings_spark.py or movie_view_ratings_beam.py.

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
from examples.example_utils import *
import pipeline_dp

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_enum('framework', None, ['beam', 'spark', 'local'],
                  'Pipeline framework to use.')
flags.DEFINE_list('public_partitions', None,
                  'List of comma-separated public partition keys')
flags.DEFINE_boolean(
    'private_partitions', False,
    'Output private partitions (do not calculate any DP metrics)')


def calculate_private_result(movie_views, pipeline_backend):
    if FLAGS.private_partitions:
        return get_private_movies(movie_views, pipeline_backend)
    else:
        return calc_dp_rating_metrics(movie_views, pipeline_backend,
                                      get_public_partitions())


def calc_dp_rating_metrics(movie_views, backend, public_partitions):
    """Computes DP metrics."""

    # Set the total privacy budget.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-6)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # Specify which DP aggregated metrics to compute.
    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[
            pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM,
            pipeline_dp.Metrics.PRIVACY_ID_COUNT
        ],
        max_partitions_contributed=2,
        max_contributions_per_partition=1,
        min_value=1,
        max_value=5,
        public_partitions=public_partitions)

    # Specify how to extract privacy_id, partition_key and value from an
    # element of movie view collection.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda mv: mv.movie_id,
        privacy_id_extractor=lambda mv: mv.user_id,
        value_extractor=lambda mv: mv.rating)

    # Run aggregation.
    dp_result = dp_engine.aggregate(movie_views, params, data_extractors)

    budget_accountant.compute_budgets()
    return dp_result


def get_private_movies(movie_views, backend):
    """Obtains the list of movies in a private manner.

    This does not calculate any private metrics; it merely obtains the list of
    movies but does so making sure the result is differentially private.
    """

    # Set the total privacy budget.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=0.1,
                                                          total_delta=1e-6)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # Specify how to extract privacy_id, partition_key and value from an
    # element of movie view collection.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda mv: mv.movie_id,
        privacy_id_extractor=lambda mv: mv.user_id)

    # Run aggregation.
    dp_result = dp_engine.select_partitions(
        movie_views,
        pipeline_dp.SelectPartitionsParams(max_partitions_contributed=2),
        data_extractors=data_extractors)

    budget_accountant.compute_budgets()
    return dp_result


def get_public_partitions():
    public_partitions = None
    if FLAGS.public_partitions is not None:
        public_partitions = [
            int(partition) for partition in FLAGS.public_partitions
        ]
    return public_partitions


def compute_on_beam():
    runner = fn_api_runner.FnApiRunner()  # local runner
    with beam.Pipeline(runner=runner) as pipeline:
        movie_views = pipeline | beam.io.ReadFromText(
            FLAGS.input_file) | beam.ParDo(ParseFile())
        pipeline_backend = pipeline_dp.BeamBackend()
        dp_result = calculate_private_result(movie_views, pipeline_backend)
        dp_result | beam.io.WriteToText(FLAGS.output_file)


def compute_on_spark():
    master = "local[1]"  # run Spark locally with one worker thread to load the input file into 1 partition
    conf = pyspark.SparkConf().setMaster(master)
    sc = pyspark.SparkContext(conf=conf)
    movie_views = sc.textFile(FLAGS.input_file) \
        .mapPartitions(parse_partition)
    pipeline_backend = pipeline_dp.SparkRDDBackend()
    dp_result = calculate_private_result(movie_views, pipeline_backend)

    delete_if_exists(FLAGS.output_file)
    dp_result.saveAsTextFile(FLAGS.output_file)


def compute_on_local_backend():
    movie_views = parse_file(FLAGS.input_file)
    pipeline_backend = pipeline_dp.LocalBackend()
    dp_result = list(calculate_private_result(movie_views, pipeline_backend))
    write_to_file(dp_result, FLAGS.output_file)


def main(unused_argv):
    if FLAGS.framework == 'beam':
        compute_on_beam()
    elif FLAGS.framework == 'spark':
        compute_on_spark()
    else:
        compute_on_local_backend()
    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
