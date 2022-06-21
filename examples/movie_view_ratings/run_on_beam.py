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
""" Demo of PipelineDP with Apache Beam.

For running:
1. Install Python and run on the command line `pip install pipeline-dp apache-beam absl-py`
2. Run python python run_on_beam.py --input_file=<path to data.txt from 3> --output_file=<...>

"""

from absl import app
# from absl import flags
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
import pipeline_dp
from pipeline_dp import private_beam
from pipeline_dp import SumParams, PrivacyIdCountParams
from pipeline_dp.private_beam import MakePrivate
from common_utils import ParseFile
#
# FLAGS = flags.FLAGS
# flags.DEFINE_string('input_file', None, 'The file with the movie view data')
# flags.DEFINE_string('output_file', None, 'Output file')


def main(unused_argv):
    # Setup Beam

    # Here, we use a local Beam runner.
    # For a truly distributed calculation, connect to a Beam cluster (e.g.
    # running on some cloud provider).
    runner = fn_api_runner.FnApiRunner()  # Local Beam runner
    with beam.Pipeline(runner=runner) as pipeline:

        # Define the privacy budget available for our computation.
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-6)

        # Load and parse input data
        movie_views_pcol = pipeline | \
                           beam.io.ReadFromText(input_file) | \
                           beam.ParDo(ParseFile())

        # Wrap Beam's PCollection into it's private version
        private_movie_views = (movie_views_pcol |
                               'Create private collection' >> MakePrivate(
                                   budget_accountant=budget_accountant,
                                   privacy_id_extractor=lambda mv: mv.user_id))

        global_params = pipeline_dp.aggregate_params.AggregationBuilderParams(
            partition_extractor=lambda mv: mv.movie_id,
            noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
            max_partitions_contributed=2,
            max_contributions_per_partition=1,
        )

        scalar_value_params = pipeline_dp.aggregate_params.ScalarValueParams(
            min_value=1, max_value=5)

        # Calculate the private sum
        dp_result = private_movie_views | \
                    "Private aggregate" >> private_beam.AggregationBuilder(global_params, [1,2,3,4,5]).\
                        aggregate_value(lambda mv:mv.rating, metrics=[pipeline_dp.Metrics.MEAN, pipeline_dp.Metrics.COUNT], output_col_name="rating1", scalar_value_params=scalar_value_params).\
                        aggregate_value(lambda mv:mv.rating, metrics=[pipeline_dp.Metrics.MEAN, pipeline_dp.Metrics.COUNT], output_col_name="rating2", scalar_value_params=scalar_value_params)
        budget_accountant.compute_budgets()

        # Save the results
        dp_result | beam.io.WriteToText(output_file)

    return 0


input_file = "/usr/local/google/home/dvadym/data/movie_views/netflix_dataset_100000.txt"
output_file = "/usr/local/google/home/dvadym/IdeaProjects/Dev/dp_100000b"

# --input_file=/usr/local/google/home/dvadym/data/movie_views/netflix_dataset_100000.txt
# --output_file=/usr/local/google/home/dvadym/IdeaProjects/dp_100000

if __name__ == '__main__':
    # flags.mark_flag_as_required("input_file")
    # flags.mark_flag_as_required("output_file")
    app.run(main)
