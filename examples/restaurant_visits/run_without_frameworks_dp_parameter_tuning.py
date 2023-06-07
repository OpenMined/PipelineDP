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
""" Demo of running PipelineDP locally, without any external data processing framework.
Shows how to use compute the best contribution bounds
(e.g. max_partitions_contributed) in a differential private way.

1. Install Python and run on the command line `pip install pipeline-dp absl-py`
2. Run python run_without_frameworks_dp_parameter_tuning.py
--input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import pipeline_dp
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', 'restaurants_week_data.csv',
                    'The file with the restaraunt visits data')
flags.DEFINE_string('output_file', None, 'Output file')


def write_to_file(col, filename):
    with open(filename, 'w') as out:
        out.write('\n'.join(sorted(map(str, col))))


def main(unused_argv):
    # Load and parse input data
    df = pd.read_csv(FLAGS.input_file)
    df.rename(inplace=True,
              columns={
                  'VisitorId': 'user_id',
                  'Time entered': 'enter_time',
                  'Time spent (minutes)': 'spent_minutes',
                  'Money spent (euros)': 'spent_money',
                  'Day': 'day'
              })
    restaraunt_visits_rows = [index_row[1] for index_row in df.iterrows()]

    # Specify how to extract privacy_id, partition_key and value from an
    # element of restaraunt_visits_rows.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda row: row.day,
        privacy_id_extractor=lambda row: row.user_id,
        value_extractor=lambda row: row.spent_money)
    # Partitions are publicly known: 7 days.
    public_partitions = list(range(1, 8))

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
        aggregation_noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        aggregation_eps=0.9,
        aggregation_delta=1e-6,
        calculation_eps=0.1,
        max_partitions_contributed_upper_bound=7)

    # Define the privacy budget available for our computation. It will be used
    # only for aggregations, therefore total_epsilon here is aggregation_eps
    # only.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(
        dp_tuning_params.aggregation_eps, dp_tuning_params.aggregation_delta)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    # Create a computational graph for the calculation.
    # All computations are lazy.
    private_contribution_bounds_col = dp_engine.calculate_private_contribution_bounds(
        restaraunt_visits_rows, dp_tuning_params, data_extractors,
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
        # only COUNT and PRIVACY_ID_COUNT are supported for as of April 2023.
        metrics=[
            pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.PRIVACY_ID_COUNT
        ],
        max_partitions_contributed=private_contribution_bounds.
        max_partitions_contributed,
        # max_contributions_per_partition are not yet implemented.
        max_contributions_per_partition=2)

    # dp_result is iterable, but iterating it would fail until budget is
    # computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.aggregate(restaraunt_visits_rows, params,
                                    data_extractors, public_partitions)

    budget_accountant.compute_budgets()

    dp_result = list(dp_result)

    # Save the results
    write_to_file(dp_result, FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("output_file")
    app.run(main)
