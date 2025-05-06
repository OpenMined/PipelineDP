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
""" Demo of running PipelineDP locally, without any external data processing framework"""

from absl import app
from absl import flags
import pipeline_dp

from common_utils import parse_file, write_to_file

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_bool(
    'pld_accounting', False, 'If false Naive budget accounting '
    'is used, if true PLD accounting')
flags.DEFINE_integer('pre_threshold', None,
                     'Pre threshold which is used in the DP aggregation')


def main(unused_argv):
    # Here, we use a local backend for computations. This does not depend on
    # any pipeline framework and it is implemented in pure Python in
    # PipelineDP. It keeps all data in memory and is not optimized for large data.
    # For datasets smaller than ~tens of megabytes, local execution without any
    # framework is faster than local mode with Beam or Spark.
    backend = pipeline_dp.LocalBackend()

    # Define the privacy budget available for our computation.
    if FLAGS.pld_accounting:
        budget_accountant = pipeline_dp.PLDBudgetAccountant(total_epsilon=1,
                                                            total_delta=1e-6)
    else:
        budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                              total_delta=1e-6)

    # Load and parse input data
    movie_views = parse_file(FLAGS.input_file)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # Define metrics to compute. We can compute multiple metrics at once.
    metrics = [
        # We can compute multiple metrics at once.
        pipeline_dp.Metrics.COUNT,
        pipeline_dp.Metrics.SUM,
        pipeline_dp.Metrics.PRIVACY_ID_COUNT
    ]
    if not FLAGS.pld_accounting:
        # PLD accounting does not yet support PERCENTILE computations.
        metrics.extend([
            pipeline_dp.Metrics.PERCENTILE(50),
            pipeline_dp.Metrics.PERCENTILE(90),
            pipeline_dp.Metrics.PERCENTILE(99)
        ])
    params = pipeline_dp.AggregateParams(
        metrics=metrics,
        # Add Gaussian noise to anonymize values.
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        # Limits to how much one user can contribute:
        # .. at most two movies rated per user
        max_partitions_contributed=2,
        # .. at most one rating for each movie
        max_contributions_per_partition=1,
        # .. with minimal rating of "1"
        min_value=1,
        # .. and maximum rating of "5"
        max_value=5,
        output_noise_stddev=True)

    if FLAGS.pre_threshold:
        params.pre_threshold = FLAGS.pre_threshold

    # Specify how to extract privacy_id, partition_key and value from an
    # element of movie_views.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda mv: mv.movie_id,
        privacy_id_extractor=lambda mv: mv.user_id,
        value_extractor=lambda mv: mv.rating)

    # Create the Explain Computation report object for passing it into
    # DPEngine.aggregate().
    explain_computation_report = pipeline_dp.ExplainComputationReport()

    # Create a computational graph for the aggregation.
    # All computations are lazy. dp_result is iterable, but iterating it would
    # fail until budget is computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.aggregate(
        movie_views,
        params,
        data_extractors,
        public_partitions=list(range(1, 100)),
        out_explain_computation_report=explain_computation_report)

    budget_accountant.compute_budgets()

    # Generate the Explain Computation report. It must be called after
    # budget_accountant.compute_budgets().
    print(explain_computation_report.text())

    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results
    dp_result = list(dp_result)

    # Save the results
    write_to_file(dp_result, FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
