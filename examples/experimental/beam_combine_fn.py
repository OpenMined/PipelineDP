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
2. Run python python beam_combine_fn.py --input_file=<path to data.txt from 3> --output_file=<...>

"""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
import pipeline_dp
from pipeline_dp import private_beam as pbeam
from examples.movie_view_ratings.common_utils import ParseFile
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')


class DPSumCombineFn(pbeam.PrivateCombineFn):

    def __init__(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value

    def create_accumulator(self):
        return 0

    def add_private_input(self, accumulator, input):
        return accumulator + np.clip(input, self._min_value, self._max_value)

    def merge_accumulators(self, accumulators):
        return sum(accumulators)

    def extract_private_output(self, accumulator):
        # Simple implementation of Laplace mechanism.
        max_abs_value = np.maximum(np.abs(self._min_value),
                                   np.abs(self._max_value))
        sensitivity = self._aggregate_params.max_contributions_per_partition * \
                      self._aggregate_params.max_partitions_contributed * max_abs_value
        eps = self._budget.eps
        laplace_b = sensitivity / eps

        # Warning: using a standard laplace noise is done only for simplicity, don't use it in production.
        # Better it's to use a standard PipelineDP metric Sum.
        return np.random.laplace(accumulator, laplace_b)

    def request_budget(self, budget_accountant):
        self._budget = budget_accountant.request_budget(
            pipeline_dp.MechanismType.LAPLACE)
        # _budget object is not initialized yet. It will be initialized with
        # eps/delta only during budget_accountant.compute_budgets() call.
        # Warning: do not access eps/delta or make deep copy of _budget object
        # in this function.


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
                           beam.io.ReadFromText(FLAGS.input_file) | \
                           beam.ParDo(ParseFile())

        # Wrap Beam's PCollection into it's private version
        private_movie_views = (movie_views_pcol |
                               'Create private collection' >> pbeam.MakePrivate(
                                   budget_accountant=budget_accountant,
                                   privacy_id_extractor=lambda mv: mv.user_id))

        private_movie_views = private_movie_views | pbeam.Map(
            lambda mv: (mv.movie_id, mv.rating))

        # Calculate the private sum
        dp_result = private_movie_views | pbeam.CombinePerKey(
            DPSumCombineFn(min_value=1, max_value=5),
            pbeam.CombinePerKeyParams(
                # Limits to how much one user can contribute:
                # .. at most two movies rated per user
                max_partitions_contributed=2,
                # .. at most one rating for each movie
                max_contributions_per_partition=1))
        budget_accountant.compute_budgets()

        # Save the results
        dp_result | beam.io.WriteToText(FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
