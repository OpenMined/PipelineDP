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
import collections

import utility_analysis_new
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
flags.DEFINE_boolean('run_on_preaggregated_data', False,
                     'If true, the data is preaggregated before tuning')


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


def preaggregate(col: list, data_extractors: pipeline_dp.DataExtractors):
    """Preaggregates a collection col.

    The output is a collection with elements
    (partition_key, (count, sum, n_partitions)).
    Each element corresponds to each (privacy_id, partition_key) which is
    present in the dataset. count and sum correspond to count and sum of values
    contributed by the privacy_key to the partition_key. n_partitions is the
    number of partitions which privacy_id contributes.
    """
    pid_pk = set((data_extractors.privacy_id_extractor(row),
                  data_extractors.partition_extractor(row)) for row in col)
    # (pid, pk)
    pid = [kv[0] for kv in pid_pk]
    # (pid,)
    pid_n_partitions = collections.Counter(pid)

    def preaggregate_fn(pk_pid_rows):
        """Aggregates rows per (partition_key, privacy_id)."""
        (pk, pid), rows = pk_pid_rows
        c = s = 0
        for row in rows:
            c += 1
            s += data_extractors.value_extractor(row)
        return (pk, (c, s, pid_n_partitions[pid]))

    backend = pipeline_dp.LocalBackend()
    key_fn = lambda row: (data_extractors.partition_extractor(row),
                          data_extractors.privacy_id_extractor(row))
    col = backend.map(col, lambda x: (key_fn(x), x))
    return list(backend.map(backend.group_by_key(col), preaggregate_fn))


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
    # Hist is 1-element iterable and the single element is a computed histogram.
    hist = list(hist)[0]

    minimizing_function = parameter_tuning.MinimizingFunction.ABSOLUTE_ERROR
    parameters_to_tune = parameter_tuning.ParametersToTune(
        max_partitions_contributed=True, max_contributions_per_partition=True)
    tune_options = parameter_tuning.TuneOptions(
        epsilon=1,
        delta=1e-5,
        aggregate_params=aggregate_params,
        function_to_minimize=minimizing_function,
        parameters_to_tune=parameters_to_tune,
        pre_aggregated_data=FLAGS.run_on_preaggregated_data)
    if FLAGS.run_on_preaggregated_data:
        input = preaggregate(restaurant_visits_rows, data_extractors)
        data_extractors = utility_analysis_new.PreAggregateExtractors(
            partition_extractor=lambda row: row[0],
            preaggregate_extractor=lambda row: row[1])
    else:
        input = restaurant_visits_rows
    if FLAGS.output_file_per_partition_analysis:
        result, per_partition = parameter_tuning.tune(
            input,
            backend,
            hist,
            tune_options,
            data_extractors,
            public_partitions,
            return_utility_analysis_per_partition=True)
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
