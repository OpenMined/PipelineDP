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
2. Run python run_on_beam.py --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions
import pipeline_dp
from pipeline_dp import private_beam
from pipeline_dp import SumParams
import pandas as pd

FLAGS = flags.FLAGS
# flags.DEFINE_string('input_file', 'restaurants_week_data.csv',
#                     'The CSV file with restauraunt visits data')
# flags.DEFINE_string('output_file', None, 'Output file')
INPUT_FILE = "gs://dataflow-apache-quickstart_handy-freedom-401310/data"
OUTPUT_FILE = "gs://dataflow-apache-quickstart_handy-freedom-401310/beam_result/result1"


def main(unused_argv):
    # Setup Beam

    # Here, we use a local Beam runner.
    # For a truly distributed calculation, connect to a Beam cluster (e.g.
    # running on some cloud provider).
    # runner = fn_api_runner.FnApiRunner()  # Local Beam runner

    options = PipelineOptions()
    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.project = 'handy-freedom-401310'  # Replace with your actual project ID
    google_cloud_options.job_name = 'pipeline_dp_restaurant'  # Replace with your desired job name
    google_cloud_options.staging_location = 'gs://dataflow-apache-quickstart_handy-freedom-401310/staging'  # Replace with your GCS staging bucket
    google_cloud_options.temp_location = 'gs://dataflow-apache-quickstart_handy-freedom-401310/temp'  # Replace with your GCS Temporary files bucket

    with beam.Pipeline(options=options) as pipeline:

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
        beam_data = pipeline | beam.Create(restaraunt_visits_rows)

        # Wrap Beam's PCollection into it's private version
        private_restaraunt_visits = beam_data | private_beam.MakePrivate(
            budget_accountant=budget_accountant,
            privacy_id_extractor=lambda row: row.user_id)

        # Calculate the private sum
        dp_result = private_restaraunt_visits | private_beam.Sum(
            SumParams(noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
                      max_partitions_contributed=7,
                      max_contributions_per_partition=2,
                      min_value=1,
                      max_value=100,
                      budget_weight=1,
                      partition_extractor=lambda row: row.day,
                      value_extractor=lambda row: row.spent_money))
        budget_accountant.compute_budgets()

        # Save the results
        dp_result | beam.io.WriteToText(OUTPUT_FILE)

    return 0


if __name__ == '__main__':
    # flags.mark_flag_as_required("input_file")
    # flags.mark_flag_as_required("output_file")
    app.run(main)
