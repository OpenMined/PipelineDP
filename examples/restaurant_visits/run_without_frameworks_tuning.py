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
"""Demo of running PipelineDP locally, without any external data processing framework.

This demo outputs a utility analysis of errors and noise for each partition in the dataset.

1. Install Python and run on the command line `pip install pipeline-dp absl-py`
2. Run python python run_without_frameworks_utility_analysis.py --output_file=<...>
"""
from absl import app
from absl import flags
import pipeline_dp
import pandas as pd

from utility_analysis_new import histograms
from utility_analysis_new import parameter_tuning

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', 'restaurants_week_data.csv',
                    'The file with the restaurant visits data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_string(
    'output_file_per_partition_analysis', None,
    'If set, partition utility analysis is output to this file')
flags.DEFINE_boolean('public_partitions', False,
                     'Whether public partitions are used')


def write_to_file(col, filename):
    with open(filename, 'w') as out:
        out.write('\n'.join(map(str, col)))


def load_data(input_file: str) -> list:
    df = pd.read_csv(input_file)
    df.rename(inplace=True,
              columns={
                  'VisitorId': 'user_id',
                  'Time entered': 'enter_time',
                  'Time spent (minutes)': 'spent_minutes',
                  'Money spent (euros)': 'spent_money',
                  'Day': 'day'
              })
    # Double the inputs so we have twice as many contributions per partition
    df_double = pd.concat([df, df])
    df_double.columns = df.columns
    return [index_row[1] for index_row in df_double.iterrows()]


def get_aggregate_params():
    # Limit contributions to 1 per partition, contribution error will be half of the count.
    return pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.GAUSSIAN,
        metrics=[pipeline_dp.Metrics.COUNT],
        max_partitions_contributed=1,
        max_contributions_per_partition=1)


def get_data_extractors():
    # Specify how to extract privacy_id, partition_key and value from an
    # element of restaurant_visits_rows.
    return pipeline_dp.DataExtractors(
        partition_extractor=lambda row: row.day,
        privacy_id_extractor=lambda row: row.user_id,
        value_extractor=lambda row: row.spent_money)


def tune_parameters():
    # Load data
    restaurant_visits_rows = load_data(FLAGS.input_file)
    # Create aggregate_params, data_extractors and public partitions.
    aggregate_params = get_aggregate_params()
    data_extractors = get_data_extractors()
    public_partitions = list(range(1, 8)) if FLAGS.public_partitions else None
    backend = pipeline_dp.LocalBackend()

    hist = histograms.compute_dataset_histograms(restaurant_visits_rows,
                                                 data_extractors, backend)
    hist = list(hist)[0]

    minimizing_function = parameter_tuning.MinimizingFunction.ABSOLUTE_ERROR
    parameters_to_tune = parameter_tuning.ParametersToTune(
        max_partitions_contributed=True, max_contributions_per_partition=True)
    tune_options = parameter_tuning.TuneOptions(
        epsilon=1,
        delta=1e-5,
        aggregate_params=aggregate_params,
        function_to_minimize=minimizing_function,
        parameters_to_tune=parameters_to_tune)
    if FLAGS.output_file_per_partition_analysis:
        result, per_partition = parameter_tuning.tune(restaurant_visits_rows,
                                                      backend, hist,
                                                      tune_options,
                                                      data_extractors,
                                                      public_partitions, True)
        write_to_file(per_partition, FLAGS.output_file_per_partition_analysis)
    else:
        result = parameter_tuning.tune(restaurant_visits_rows, backend, hist,
                                       tune_options, data_extractors,
                                       public_partitions, False)

    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results
    result = list(result)

    # Save the results
    write_to_file(result, FLAGS.output_file)


def main(unused_args):
    tune_parameters()
    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("output_file")
    app.run(main)
