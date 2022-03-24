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

4. Run python custom_combiners.py --framework=<framework> --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
from apache_beam.runners.portability import fn_api_runner
import pyspark
from examples.movie_view_ratings.common_utils import *
import pipeline_dp
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_enum('framework', None, ['beam', 'spark', 'local'],
                  'Pipeline framework to use.')
flags.DEFINE_list('public_partitions', None,
                  'List of comma-separated public partition keys')


def calculate_private_result(movie_views, pipeline_backend):
    return calc_dp_rating_metrics(movie_views, pipeline_backend,
                                  get_public_partitions())


class CountCombiner(pipeline_dp.CustomCombiner):
    """DP sum combiner.

    It is just for demonstration how custom combiners API work.
    """

    def create_accumulator(self, values):
        """Creates accumulator from 'values'."""
        return len(values)

    def merge_accumulators(self, count1, count2):
        """Merges the accumulators and returns accumulator."""
        return count1 + count2

    def compute_metrics(self, count):
        """Computes and returns the result of aggregation."""
        # Simple implementation of Laplace mechanism.
        sensitivity = self._aggregate_params.max_contributions_per_partition * \
                      self._aggregate_params.max_partitions_contributed
        eps = self._budget.eps
        laplace_b = sensitivity / eps

        # Warning: using a standard laplace noise is done only for simplicity, don't use it in production.
        # Better it's to use a standard PipelineDP metric Count.
        return np.random.laplace(count, laplace_b)

    def request_budget(self, budget_accountant):
        self._budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.LAPLACE)
        # _budget object is not initialized yet. It will be initialized with
        # eps/delta only during budget_accountant.compute_budgets() call.
        # Warning: do not access eps/delta or make deep copy of _budget object
        # in this function.

    def set_aggregate_params(self, aggregate_params):
        self._aggregate_params = aggregate_params


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
        metrics=None,
        max_partitions_contributed=2,
        max_contributions_per_partition=1,
        min_value=1,
        max_value=5,
        public_partitions=public_partitions,
        custom_combiners=[CountCombiner()])

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
