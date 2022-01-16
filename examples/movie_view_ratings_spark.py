""" Demo of PipelineDP with Spark.
"""

from absl import app
from absl import flags
import pyspark
import pipeline_dp
from pipeline_dp.private_spark import make_private
from pipeline_dp import SumParams
from examples.example_utils import parse_partition
from examples.example_utils import delete_if_exists

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')


def main(unused_argv):
    # Setup Spark
    master = "local[1]"  # use one worker thread to load the file as 1 partition
    conf = pyspark.SparkConf().setMaster(master)
    sc = pyspark.SparkContext(conf=conf)
    movie_views = sc \
        .textFile(FLAGS.input_file) \
        .mapPartitions(parse_partition)

    # Define the privacy budget available for our computation.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-6)

    # Wrap Spark's RDD into it's private version
    private_movie_views = \
        make_private(movie_views, budget_accountant, lambda mv: mv.user_id)

    # Calculate the private sum
    dp_result = private_movie_views.sum(
        SumParams(max_partitions_contributed=2,
                  max_contributions_per_partition=2,
                  low=1,
                  high=5,
                  partition_extractor=lambda mv: mv.movie_id,
                  value_extractor=lambda mv: mv.rating))

    budget_accountant.compute_budgets()

    # Save the results
    delete_if_exists(FLAGS.output_file)
    dp_result.saveAsTextFile(FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
