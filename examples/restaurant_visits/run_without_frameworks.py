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
2. Run python python run_without_frameworks.py --input_file=<path to data.txt from 3> --output_file=<...>
"""

from absl import app
from absl import flags
import pipeline_dp
import pandas as pd
import pyspark
import os
import shutil
from pyspark.sql import SparkSession

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', 'restaurants_week_data.csv',
                    'The file with the restaraunt visits data')
flags.DEFINE_string('output_file', None, 'Output file')

RUN_ON_SPARK = True


def delete_if_exists(filename):
    if os.path.exists(filename):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)


def write_to_file(col, filename):
    with open(filename, 'w') as out:
        out.write('\n'.join(map(str, col)))


def get_spark_context():
    if not RUN_ON_SPARK:
        return None
    master = "local[1]"  # use one worker thread to load the file as 1 partition
    #Create PySpark SparkSession
    spark = SparkSession.builder \
      .master("local[1]") \
      .appName("SparkByExamples.com") \
      .getOrCreate()
    return spark
    # conf = pyspark.SparkConf().setMaster(master)
    #   return pyspark.SparkContext(conf=conf)


def get_data(sc):
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
    if not RUN_ON_SPARK:
        return df

    spark_df = sc.createDataFrame(df)
    return spark_df


def get_backend(sc):
    if RUN_ON_SPARK:
        return pipeline_dp.pipeline_backend.SparkDataFrameBackend(sc)
    return pipeline_dp.pipeline_backend.PandasDataFrameBackend(sc)


def main(unused_argv):
    # Silence some Spark warnings
    import warnings
    warnings.simplefilter('ignore', UserWarning)
    warnings.simplefilter('ignore', ResourceWarning)
    delete_if_exists(FLAGS.output_file)
    # Here, we use a local backend for computations. This does not depend on
    # any pipeline framework and it is implemented in pure Python in
    # PipelineDP. It keeps all data in memory and is not optimized for large data.
    # For datasets smaller than ~tens of megabytes, local execution without any
    # framework is faster than local mode with Beam or Spark.
    # backend = pipeline_dp.LocalBackend()
    sc = get_spark_context()
    backend = get_backend(sc)

    # Define the privacy budget available for our computation.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-6)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    params = pipeline_dp.AggregateParams(
        noise_kind=pipeline_dp.NoiseKind.LAPLACE,
        metrics=[pipeline_dp.Metrics.COUNT, pipeline_dp.Metrics.SUM],
        max_partitions_contributed=3,
        max_contributions_per_partition=2,
        min_value=0,
        max_value=60)

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
    df = get_data(sc)
    dp_result = dp_engine.aggregate(df,
                                    params,
                                    data_extractors,
                                    public_partitions=list(range(1, 8)))

    budget_accountant.compute_budgets()
    df = dp_result.collect()
    # dp_result_df = sc.createDataFrame(dp_result)

    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results
    # dp_result = list(dp_result_df)

    # Save the results
    write_to_file(dp_result, FLAGS.output_file)

    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("output_file")
    app.run(main)
