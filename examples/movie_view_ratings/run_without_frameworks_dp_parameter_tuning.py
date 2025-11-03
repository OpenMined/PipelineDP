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
""" Demo of running PipelineDP locally, without any external
data processing framework"""

from absl import app
from absl import flags
import pipeline_dp

from examples.movie_view_ratings.common_utils import parse_file, write_to_file
from examples.movie_view_ratings import common_utils

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')


def main(unused_argv):
    # Load and parse input data
    movie_views = parse_file(FLAGS.input_file)

    # Specify how to extract privacy_id, partition_key and value from an
    # element of movie_views.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda mv: mv.movie_id,
        privacy_id_extractor=lambda mv: mv.user_id,
        value_extractor=lambda mv: mv.rating)
    # There is no support for private partition selection without specifying
    # max_partitions_contributed therefore we treat partitions as public.
    public_partitions = common_utils.get_partitions(
        data_extractors.partition_extractor, movie_views)

    # LocalBackend is more efficient for small datasets.
    backend = pipeline_dp.LocalBackend()

    # For private calculation of contribution bounds we need to decide ahead
    # of time how much privacy budget we will spend on the actual aggregation
    # and how much privacy budget we will spend on the parameter tuning. Also,
    # we need to understand what type of noise we will use for aggregations.
    # In this example, we use total epsilon budget 1 and split it
    # 0.9 on aggregation and 0.1 on calculation. Delta is not needed for DP
    # bounds calculations, therefore we don't split it.
    dp_tuning_params = pipeline_dp.CalculatePrivateContributionBoundsParams(
        aggregation_noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        aggregation_eps=0.9,
        aggregation_delta=1e-6,
        calculation_eps=0.1,
        max_partitions_contributed_upper_bound=100)

    # Define the privacy budget available for our computation. It will be used
    # only for aggregations, therefore total_epsilon here is aggregation_eps
    # only.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        dp_tuning_params.aggregation_eps, dp_tuning_params.aggregation_delta)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # Create a computational graph for the calculation.
    # All computations are lazy.
    private_contribution_bounds_col = \
        dp_engine.calculate_private_contribution_bounds(
        movie_views, dp_tuning_params, data_extractors,
        public_partitions)
    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results.
    private_contribution_bounds: pipeline_dp.PrivateContributionBounds = list(
        private_contribution_bounds_col)[0]

    print(
        f"Calculated private contribution bounds: {private_contribution_bounds}"
    )

    params = pipeline_dp.AggregateParams(
        noise_kind=dp_tuning_params.aggregation_noise_kind,
        # only COUNT and PRIVACY_ID_COUNT are supported for now/
        metrics=[
            # we can compute multiple metrics at once.
            # For DP computation of contribution bounds only COUNT and
            # PRIVACY_ID_COUNT is support as of April 2023.
            pipeline_dp.Metrics.COUNT,
            pipeline_dp.Metrics.PRIVACY_ID_COUNT,
        ],
        # Limits to how much one user can contribute:
        # .. at most two movies rated per user
        # we estimated the best value for it in a dp way above.
        max_partitions_contributed=private_contribution_bounds.
        max_partitions_contributed,
        # .. at most one rating for each movie
        # DP estimation for max_contributions_per_partition is not yet
        # supported.
        max_contributions_per_partition=1)

    # Create the Explain Computation report object for passing it into
    # DPEngine.aggregate().
    explain_computation_report = pipeline_dp.ExplainComputationReport()

    # Create a computational graph for the aggregation.
    # All computations are lazy. dp_result is iterable, but iterating it would
    # fail until budget is computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.aggregate(movie_views, params, data_extractors,
                                    public_partitions,
                                    explain_computation_report)

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


