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
"""Helper utilities used by examples.

Prefer not to include any privacy-related logic here. Keep privacy related code
in the actual example scripts.
"""
from dataclasses import dataclass
import os
import shutil
import apache_beam as beam


@dataclass
class MovieView:
    user_id: int
    movie_id: int
    rating: int


def parse_line(line, movie_id):
    # 'line' has format "user_id,rating,date"
    split_parts = line.split(',')
    user_id = int(split_parts[0])
    rating = int(split_parts[1])
    return MovieView(user_id, movie_id, rating)


def parse_partition(iterator):
    movie_id = None
    for line in iterator:
        if line[-1] == ':':
            # 'line' has a format "movie_id:'
            movie_id = int(line[:-1])
        else:
            # 'line' has a format "user_id,rating,date"
            yield parse_line(line, movie_id)


def parse_file(filename):  # used for the local run
    res = []
    for line in open(filename):
        line = line.strip()
        if line[-1] == ':':
            movie_id = int(line[:-1])
        else:
            res.append(parse_line(line, movie_id))
    return res


def write_to_file(col, filename):
    with open(filename, 'w') as out:
        out.write('\n'.join(map(str, col)))


def delete_if_exists(filename):
    if os.path.exists(filename):
        if os.path.isdir(filename):
            shutil.rmtree(filename)
        else:
            os.remove(filename)


class ParseFile(beam.DoFn):

    def __init__(self):
        self.movie_id = -1

    def process(self, line):
        if line[-1] == ':':
            # 'line' has a format "movie_id:'
            self.movie_id = int(line[:-1])
            return
        # 'line' has a format "user_id,rating,date"
        yield parse_line(line, self.movie_id)
