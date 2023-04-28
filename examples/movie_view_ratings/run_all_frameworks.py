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
""" The example of using DPEngine for performing DP aggregation.

This is a quite elaborate example demonstrating many features. For a simpler
example of how to use PipelineDP with spark, check
run_on_spark.py or run_on_beam.py.

In order to run an example:

1. Install Python and run on the command line `pip install pipeline-dp apache-beam pyspark absl-py`
2. Download the Netflix prize dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data and unpack it.
3. The dataset itself is pretty big, to speed up the run it's better to use a
part of it. You can get a part of it by running in bash:

   head -10000 combined_data_1.txt > data.txt

4. Run python run_all_frameworks.py --framework=<framework> --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
from apache_beam.runners.portability import fn_api_runner
import pyspark
from examples.movie_view_ratings.common_utils import *
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
flags.DEFINE_boolean(
    'contribution_bounds_already_enforced', False,
    'Assume the input dataset already enforces the hard-coded contribution'
    'bounds. Ignore the user identifiers.')
flags.DEFINE_boolean('vector_metrics', False,
                     'Compute DP vector metrics for rating values')


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

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[
            pipeline_dp.Metrics.COUNT,
            pipeline_dp.Metrics.SUM,
            # pipeline_dp.Metrics.MEAN, pipeline_dp.Metrics.VARIANCE
        ] + ([pipeline_dp.Metrics.PRIVACY_ID_COUNT]
             if not FLAGS.contribution_bounds_already_enforced else []),
        max_partitions_contributed=2,
        max_contributions_per_partition=1,
        min_value=1,
        max_value=5,
        contribution_bounds_already_enforced=FLAGS.
        contribution_bounds_already_enforced)

    value_extractor = lambda mv: mv.rating

    if FLAGS.vector_metrics:
        # Specify which DP aggregated metrics to compute for vector values.
        params.metrics = [pipeline_dp.Metrics.VECTOR_SUM]
        params.vector_size = 5  # Size of ratings vector
        params.vector_max_norm = 1
        value_extractor = lambda mv: encode_one_hot(mv.rating - 1, params.
                                                    vector_size)

    # Specify how to extract privacy_id, partition_key and value from an
    # element of movie view collection.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda mv: mv.movie_id,
        privacy_id_extractor=(lambda mv: mv.user_id)
        if not FLAGS.contribution_bounds_already_enforced else None,
        value_extractor=value_extractor)

    # Run aggregation.
    dp_result = dp_engine.aggregate(movie_views, params, data_extractors,
                                    public_partitions)

    budget_accountant.compute_budgets()

    reports = dp_engine.explain_computations_report()
    for report in reports:
        print(report)

    return dp_result


def get_private_movies(movie_views, backend):
    """Obtains the list of movies in a differentially private manner.

    This does not calculate any metrics; it merely returns the list of
    movies, making sure the result is differentially private.
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
    pipeline_backend = pipeline_dp.SparkRDDBackend(sc)
    dp_result = calculate_private_result(movie_views, pipeline_backend)

    delete_if_exists(FLAGS.output_file)
    dp_result.saveAsTextFile(FLAGS.output_file)


def compute_on_local_backend():
    movie_views = parse_file(FLAGS.input_file)
    pipeline_backend = pipeline_dp.LocalBackend()
    dp_result = list(calculate_private_result(movie_views, pipeline_backend))
    write_to_file(dp_result, FLAGS.output_file)


def encode_one_hot(value, vector_size):
    vec = [0] * vector_size
    vec[value] = 1
    return vec


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
