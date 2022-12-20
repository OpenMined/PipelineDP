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
"""Pre aggregation Test"""
from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp
import analysis


class PreaggregationTests(parameterized.TestCase):

    @parameterized.named_parameters(
        dict(testcase_name="Empty dataset", input=[], expected_output=[]),
        dict(
            testcase_name="Small input dataset",
            input=[
                ("pid0", "pk0", 1),
                ("pid0", "pk1", 10),
                ("pid0", "pk1", 5),
            ],
            # element format (partition_key, (count, sum, n_partitions)
            expected_output=[('pk0', (1, 1, 2)), ('pk1', (2, 15, 2))]),
        dict(
            testcase_name="10 privacy ids where each privacy id contributes to "
            "the same 10 partitions, 3 times in each partition",
            input=[(i, j, 0) for i in range(10) for j in range(10)] * 3,
            # element format (partition_key, (count, sum, n_partitions)
            expected_output=[(i, (3, 0, 10)) for i in range(10)] * 10),
    )
    def test_preaggregate(self, input, expected_output):
        data_extractors = pipeline_dp.DataExtractors(
            privacy_id_extractor=lambda row: row[0],
            partition_extractor=lambda row: row[1],
            value_extractor=lambda row: row[2])

        output = list(
            analysis.preaggregate(input, pipeline_dp.LocalBackend(),
                                  data_extractors))

        self.assertSequenceEqual(output, expected_output)


if __name__ == '__main__':
    absltest.main()
