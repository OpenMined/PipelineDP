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
""" Demo of running PipelineDP on (Pandas, Spark, Beam) DataFrames

1. Install Python and run on the command line `pip install pipeline-dp absl-py`
2. Run python run_on_dataframes.py --input_file=<path to data.txt from 3> --output_file=<...> --dataframes=pandas<spark>
"""

from absl import app
from absl import flags
import os
import shutil
import pandas as pd

from pyspark.sql import SparkSession
import pyspark

import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from apache_beam.dataframe.io import read_csv

import pipeline_dp
from pipeline_dp import dataframes

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', 'restaurants_week_data.csv',
                    'The file with the restaraunt visits data')
flags.DEFINE_string('output_file', None, 'Output file')
flags.DEFINE_enum('dataframes', 'pandas', ['pandas', 'spark', 'beam'],
                  'Which dataframes to use.')


def delete_if_exists(filename):
    if os.path.exists(filename):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)


def load_data_in_pandas_dataframe() -> pd.DataFrame:
    df = pd.read_csv(FLAGS.input_file)
    df.rename(inplace=True,
              columns={
                  'VisitorId': 'visitor_id',
                  'Time entered': 'enter_time',
                  'Time spent (minutes)': 'spent_minutes',
                  'Money spent (euros)': 'spent_money',
                  'Day': 'day'
              })
    return df


def load_data_in_spark_dataframe(
        spark: SparkSession) -> pyspark.sql.dataframe.DataFrame:
    df = spark.read.csv(FLAGS.input_file, header=True, inferSchema=True)
    return df.withColumnRenamed('VisitorId', 'visitor_id').withColumnRenamed(
        'Time entered', 'enter_time').withColumnRenamed(
            'Time spent (minutes)', 'spent_minutes').withColumnRenamed(
                'Money spent (euros)',
                'spent_money').withColumnRenamed('Day', 'day')


def load_data_in_beam_dataframe(pipeline):
    df = pipeline | read_csv(FLAGS.input_file)
    df = df.rename(
        columns={
            'VisitorId': 'visitor_id',
            'Time entered': 'enter_time',
            'Time spent (minutes)': 'spent_minutes',
            'Money spent (euros)': 'spent_money',
            'Day': 'day'
        })
    return df


def compute_private_result(df):
    dp_query_builder = dataframes.QueryBuilder(df, 'visitor_id')
    query = dp_query_builder.groupby('day', 3, 1).count().sum(
        'spent_money', min_value=0, max_value=100).build_query()
    result_df = query.run_query(dataframes.Budget(epsilon=5, delta=1e-10),
                                noise_kind=pipeline_dp.NoiseKind.GAUSSIAN)
    print(query.explain_computations())
    return result_df


def compute_on_pandas_dataframes() -> None:
    df = load_data_in_pandas_dataframe()
    result_df = compute_private_result(df)
    result_df.to_csv(FLAGS.output_file)


def compute_on_spark_dataframes() -> None:
    spark = SparkSession.builder \
      .master("local[1]") \
      .appName("SparkByExamples.com") \
      .getOrCreate()
    df = load_data_in_spark_dataframe(spark)
    df.printSchema()
    result_df = compute_private_result(df)
    result_df.printSchema()
    delete_if_exists(FLAGS.output_file)
    result_df.write.format("csv").option("header", True).save(FLAGS.output_file)


def compute_on_beam_dataframes() -> None:
    with beam.Pipeline(runner=fn_api_runner.FnApiRunner()) as pipeline:
        df = load_data_in_beam_dataframe(pipeline)
        result_df = compute_private_result(df)
        result_df.to_csv(FLAGS.output_file)


def main(unused_argv):
    if FLAGS.dataframes == 'pandas':
        compute_on_pandas_dataframes()
    elif FLAGS.dataframes == 'spark':
        compute_on_spark_dataframes()
    elif FLAGS.dataframes == 'beam':
        compute_on_beam_dataframes()
    return 0


if __name__ == '__main__':
    flags.mark_flag_as_required("output_file")
    app.run(main)
