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
""" Demo of PipelineDP with Spark.

For running:
1. Install Python and run on the command line `pip install pipeline-dp pyspark absl-py`
2. Run python python run_on_beam.py --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import pyspark
import pipeline_dp
from pipeline_dp.private_spark import make_private
from pipeline_dp import SumParams
from common_utils import parse_partition
from common_utils import delete_if_exists

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')


def main(unused_argv):
    delete_if_exists(FLAGS.output_file)

    # Setup Spark

    # Here, we use one worker thread to load the file as 1 partition.
    # For a truly distributed calculation, connect to a Spark cluster (e.g.
    # running on some cloud provider).
    master = "local[1]"  # use one worker thread to load the file as 1 partition
    conf = pyspark.SparkConf().setMaster(master)
    sc = pyspark.SparkContext(conf=conf)
    movie_views = sc \
        .textFile(FLAGS.input_file) \
        .mapPartitions(parse_partition)

    # Define the privacy budget available for our computation.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-6)

    # Wrap Spark's RDD into its private version
    private_movie_views = \
        make_private(movie_views, budget_accountant, lambda mv: mv.user_id)

    # Calculate the private sum
    dp_result = private_movie_views.sum(
        SumParams(
            # Limits to how much one user can contribute:
            # .. at most two movies rated per user
            max_partitions_contributed=2,
            # .. at most one rating for each movie
            max_contributions_per_partition=1,
            # .. with minimal rating of "1"
            min_value=1,
            # .. and maximum rating of "5"
            max_value=5,
            # The aggregation key: we're grouping by movies
            partition_extractor=lambda mv: mv.movie_id,
            # The value we're aggregating: we're summing up ratings
            value_extractor=lambda mv: mv.rating))

    budget_accountant.compute_budgets()

    # Save the results
    dp_result.saveAsTextFile(FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
