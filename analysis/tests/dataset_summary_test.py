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
"""Tests for the dataset summary."""

from absl.testing import absltest
from absl.testing import parameterized

from analysis import dataset_summary
import pipeline_dp


class PublicDatasetSummaryTest(parameterized.TestCase):

    def test_compute_public_partitions_summary(self):
        dataset = list(range(100))
        public_partitions = list(range(60, 121))
        extractors = pipeline_dp.DataExtractors(
            partition_extractor=lambda x: x, privacy_id_extractor=lambda _: 0)

        summary = dataset_summary.compute_public_partitions_summary(
            dataset, pipeline_dp.LocalBackend(), extractors, public_partitions)

        summary = list(summary)[0]

        self.assertEqual(summary.num_dataset_public_partitions, 40)
        self.assertEqual(summary.num_dataset_non_public_partitions, 60)
        self.assertEqual(summary.num_empty_public_partitions, 21)


if __name__ == '__main__':
    absltest.main()
