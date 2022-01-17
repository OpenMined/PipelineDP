import datetime as dt
import os
import time

from absl import app
from absl import flags

import apache_beam as beam
from apache_beam.runners.portability import fn_api_runner
from budget_accounting import BudgetAccountant
from dataclasses import dataclass
from private_beam import DPEngine, BeamBackend, LocalBackend, DataExtractors, AggregateParams, Metrics

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', None, 'Input/output directory')
flags.DEFINE_string('input_file', None,
                    'The file with the data, it should be in data_dir')
# False means the run based on LocalBackend(), i.e. w/o any frameworks, true local run with Apache Beam framework
flags.DEFINE_bool('use_beam', False,
                  'Set to use Beam backend, otherwise use local backend')


@dataclass
class MovieView:
  user_id: int
  movie_id: int
  rating: int
  data: dt.datetime


def parse_line(line, movie_id):
  # this line has format "user_id,rating,date"
  split_parts = line.split(',')
  user_id = int(split_parts[0])
  rating = int(split_parts[1])
  date = dt.datetime.strptime(split_parts[2], '%Y-%m-%d')
  return MovieView(user_id, movie_id, rating, date)


class ParseFile(beam.DoFn):

  def __init__(self):
    self.movie_id = -1

  def process(self, line):
    if line[-1] == ':':  # this line has format "movie_id:"
      self.movie_id = int(line[:-1])
      return
    # this line has format "user_id,rating,date"
    yield parse_line(line, self.movie_id)


def parse_file(filename):  # used for the local run
  res = []
  for line in open(filename):
    line = line.strip()
    if line[-1] == ':':
      movie_id = int(line[:-1])
    else:
      res.append(parse_line(line, movie_id))
  return res


def get_netflix_dataset(pipeline, use_beam):
  filename = os.path.join(FLAGS.data_dir, FLAGS.input_file)
  if use_beam:
    return pipeline | beam.io.ReadFromText(filename) | beam.ParDo(ParseFile())
  return parse_file(filename)


def write_to_local_file(col, filename):  # used for the local run
  if col is None:
    return
  with open(filename, 'w') as out:
    out.write('\n'.join(map(str, col)))


def calc_rating(pipeline, use_beam):
  movie_views = get_netflix_dataset(pipeline, use_beam)

  data_extractors = DataExtractors(
      partition_extractor=lambda mv: mv.movie_id,
      privacy_id_extractor=lambda mv: mv.user_id,
      value_extractor=lambda mv: mv.rating)
  params = AggregateParams(
      max_partitions_contributed=2,
      max_contributions_per_partition=1,
      low=1,
      high=5,
      metrics=[Metrics.PRIVACY_ID_COUNT, Metrics.COUNT, Metrics.MEAN],
      preagg_partition_selection=True,
      # public_partitions = list(range(1, 40))  # uncomment this line for using public partitions.
  )

  budget_accountant = BudgetAccountant(eps=1, delta=1e-6)
  ops = BeamBackend() if use_beam else LocalBackend()
  dpe = DPEngine(budget_accountant, ops)

  dp_result = dpe.aggregate(movie_views, params, data_extractors)
  budget_accountant.compute_budgets()

  # Print DP aggregation reports
  reports = dpe._report_generators
  print(f'There were {len(reports)} computations')
  for i, report in enumerate(reports):
    print(f'Computation {i}:')
    print(report.report())

  return dp_result


def compute_on_beam(private_outfile):
  runner = fn_api_runner.FnApiRunner()  # local runner
  with beam.Pipeline(runner=runner) as pipeline:
    private = calc_rating(pipeline, use_beam=True)
    private | 'private data save' >> beam.io.WriteToText(private_outfile)


def compute_locally(private_outfile):
  private = calc_rating(pipeline=None, use_beam=False)
  write_to_local_file(private, private_outfile)


def main(unused_argv):
  private_outfile = os.path.join(FLAGS.data_dir, 'dp_aggregation_result')
  starttime = time.time()
  if FLAGS.use_beam:
    compute_on_beam(private_outfile)
  else:
    compute_locally(private_outfile)
  print(f'DP aggregation running time {time.time() - starttime} seconds')
  return 0


if __name__ == '__main__':
  app.run(main)
