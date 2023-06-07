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
import pipeline_dp
from pipeline_dp import partition_selection

from absl.testing import absltest
from absl.testing import parameterized
from unittest.mock import patch


class PartitionSelectionTest(parameterized.TestCase):

    @patch("pydp.algorithms.partition_selection.create_partition_strategy")
    def test_truncated_geometric(self, mock_method):
        eps, delta, max_partitions = 2, 1e-3, 10
        partition_selection.create_partition_selection_strategy(
            pipeline_dp.PartitionSelectionStrategy.TRUNCATED_GEOMETRIC,
            eps,
            delta,
            max_partitions,
            pre_threshold=None)
        mock_method.assert_called_once_with("truncated_geometric", eps, delta,
                                            max_partitions)

    @patch("pydp.algorithms.partition_selection.create_partition_strategy")
    def test_truncated_laplace_thresholding(self, mock_method):
        eps, delta, max_partitions = 5, 1e-2, 12
        partition_selection.create_partition_selection_strategy(
            pipeline_dp.PartitionSelectionStrategy.LAPLACE_THRESHOLDING,
            eps,
            delta,
            max_partitions,
            pre_threshold=None)
        mock_method.assert_called_once_with("laplace", eps, delta,
                                            max_partitions)

    @patch("pydp.algorithms.partition_selection.create_partition_strategy")
    def test_truncated_gaussian_thresholding(self, mock_method):
        eps, delta, max_partitions = 1, 1e-5, 20
        partition_selection.create_partition_selection_strategy(
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING,
            eps,
            delta,
            max_partitions,
            pre_threshold=None)
        mock_method.assert_called_once_with("gaussian", eps, delta,
                                            max_partitions)

    @patch("pydp.algorithms.partition_selection.create_partition_strategy")
    def test_truncated_pre_thresholding(self, mock_method):
        eps, delta, max_partitions, pre_threshold = 1, 1e-5, 20, 42
        partition_selection.create_partition_selection_strategy(
            pipeline_dp.PartitionSelectionStrategy.GAUSSIAN_THRESHOLDING, eps,
            delta, max_partitions, pre_threshold)
        mock_method.assert_called_once_with("gaussian", eps, delta,
                                            max_partitions, pre_threshold)


if __name__ == '__main__':
    absltest.main()
