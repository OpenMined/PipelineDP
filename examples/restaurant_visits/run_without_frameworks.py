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
""" Demo of running PipelineDP locally, without any external data processing framework

1. Install Python and run on the command line `pip install pipeline-dp absl-py`
2. Run python run_without_frameworks.py --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import pipeline_dp
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', 'restaurants_week_data.csv',
                    'The file with the restaraunt visits data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_boolean('post_aggregation_thresholding', False,
                     'Whether post aggregation thresholding is used')
flags.DEFINE_boolean('public_partitions', False,
                     'Whether public partitions are used')


def write_to_file(col, filename):
    with open(filename, 'w') as out:
        out.write('\n'.join(map(str, col)))


def main(unused_argv):
    # Here, we use a local backend for computations. This does not depend on
    # any pipeline framework and it is implemented in pure Python in
    # PipelineDP. It keeps all data in memory and is not optimized for large data.
    # For datasets smaller than ~tens of megabytes, local execution without any
    # framework is faster than local mode with Beam or Spark.
    backend = pipeline_dp.LocalBackend()

    # Define the privacy budget available for our computation.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-6)

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

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[
            pipeline_dp.Metrics.PRIVACY_ID_COUNT, pipeline_dp.Metrics.COUNT,
            pipeline_dp.Metrics.SUM
        ],
        max_partitions_contributed=3,
        max_contributions_per_partition=2,
        min_value=0,
        max_value=60,
        post_aggregation_thresholding=FLAGS.post_aggregation_thresholding)

    # Specify how to extract privacy_id, partition_key and value from an
    # element of restaraunt_visits_rows.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda row: row.day,
        privacy_id_extractor=lambda row: row.user_id,
        value_extractor=lambda row: row.spent_money)

    # Create the Explain Computation report object for passing it into
    # DPEngine.aggregate().
    explain_computation_report = pipeline_dp.ExplainComputationReport()

    public_partitions = list(range(1, 8)) if FLAGS.public_partitions else None

    # Create a computational graph for the aggregation.
    # All computations are lazy. dp_result is iterable, but iterating it would
    # fail until budget is computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.aggregate(
        restaraunt_visits_rows,
        params,
        data_extractors,
        public_partitions=public_partitions,
        out_explain_computation_report=explain_computation_report)

    budget_accountant.compute_budgets()

    print(explain_computation_report.text())

    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results
    dp_result = list(dp_result)

    # Save the results
    write_to_file(dp_result, FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("output_file")
    app.run(main)
