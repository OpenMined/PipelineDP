# Copyright 2023 OpenMined.
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
"""This demo applies differential privacy to data that has already been
pre-aggregated. This relies on assumptions that the partitions keys are public,
and that the sensitivities of the pre-aggregated values are known.

1. Install Python and run on the command line `pip install pipeline-dp absl-py`
2. Run python run_on_preaggregated_data.py --input_file=<path to restaurants_week_data.csv> --output_file=<...>
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
    df["count"] = 0
    df_day_count = df[["day", "count"]].groupby("day").count()
    day_count = [(row[0], row[1]["count"]) for row in df_day_count.iterrows()]
    # day_count has data [(1, 300), ...]

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AddDPNoiseParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        # there are data for 7 days (partitions), so each visitor can contribute
        # to not more than to 7 partitions
        l0_sensitivity=7,
        # Assume visitor can not go to more than 5 restaurants per day.
        linf_sensitivity=5)

    # A report that helps to explain computations in the pipeline.
    explain_computation_report = pipeline_dp.ExplainComputationReport()

    # Create a computational graph for the aggregation.
    # All computations are lazy. dp_result is iterable, but iterating it would
    # fail until budget is computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.add_dp_noise(
        day_count,
        params,
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
