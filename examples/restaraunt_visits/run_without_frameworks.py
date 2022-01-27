""" Demo of running PipelineDP locally, without any external data processing framework"""

from absl import app
from absl import flags
import pipeline_dp
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', 'restaraunts_week_data.csv',
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
    restaraunt_visits_rows = [index_row[1] for index_row in df.iterrows()]

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
        max_partitions_contributed=3,
        max_contributions_per_partition=2,
        min_value=0,
        max_value=60,
        public_partitions=list(range(1, 8)))

    # Specify how to extract privacy_id, partition_key and value from an
    # element of restaraunt_visits_rows.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda row: row.day,
        privacy_id_extractor=lambda row: row.user_id,
        value_extractor=lambda row: row.spent_money)

    # Create a computational graph for the aggregation.
    # All computations are lazy. dp_result is iterable, but iterating it would
    # fail until budget is computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.aggregate(restaraunt_visits_rows, params,
                                    data_extractors)

    budget_accountant.compute_budgets()

    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results
    dp_result = list(dp_result)

    # Save the results
    write_to_file(dp_result, FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("output_file")
    app.run(main)
