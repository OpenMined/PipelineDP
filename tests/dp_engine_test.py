from collections import defaultdict
import numpy as np
import unittest

import pipeline_dp

class DPEngineTest(unittest.TestCase):

  aggregator_fn = lambda inp_values : (len(inp_values),
                                           np.sum(inp_values),
                                           np.sum(np.square(inp_values)))

  def testContributionBounding_boundInput_nothingDropped(self):
      inp_col = [("pid1", 'pk1' , 1),
                 ("pid1", 'pk1' , 2),
                 ("pid1", 'pk2' , 3),
                 ("pid1", 'pk2' , 4)]
      max_partitions_contributed = 2
      max_contributions_per_partition = 2

      dpEngine = pipeline_dp.DPEngine(pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
      pipeline_dp.LocalPipelineOperations())
      bound_op = dpEngine._bound_cross_partition_contributions(
                            inp_col,
                            max_partitions_contributed=max_partitions_contributed,
                            max_contributions_per_partition=max_contributions_per_partition,
                            aggregator_fn=DPEngineTest.aggregator_fn)

      expected_op = [(('pid1', 'pk2'), (2, 7, 25)), (('pid1', 'pk1'), (2, 3, 5))]
      self.assertEqual(set(expected_op), set(bound_op))

  def testContributionBounding_perPartitionBoundingApplied(self):
    inp_col = [("pid1", 'pk1' , 1),
               ("pid1", 'pk1' , 2),
               ("pid1", 'pk2' , 3),
               ("pid1", 'pk2' , 4),
               ("pid1", 'pk2' , 5),
               ("pid2", 'pk2' , 6)]
    max_partitions_contributed = 5
    max_contributions_per_partition = 2

    dpEngine = pipeline_dp.DPEngine(pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
    pipeline_dp.LocalPipelineOperations())
    bound_op = dpEngine._bound_cross_partition_contributions(
                          inp_col,
                          max_partitions_contributed=max_partitions_contributed,
                          max_contributions_per_partition=max_contributions_per_partition,
                          aggregator_fn=DPEngineTest.aggregator_fn)

    self.assertEqual(len(bound_op), 3)
    self.assertTrue(all(map(lambda op_val : op_val[1][0] <= max_contributions_per_partition, bound_op)))

  def testContributionBounding_crossPartitionBoundingApplied(self):
    inp_col = [("pid1", 'pk1' , 1),
               ("pid1", 'pk1' , 2),
               ("pid1", 'pk2' , 3),
               ("pid1", 'pk2' , 4),
               ("pid1", 'pk2' , 5),
               ("pid1", 'pk3' , 6),
               ("pid1", 'pk4' , 7),
               ("pid2", 'pk4' , 8)]
    max_partitions_contributed = 3
    max_contributions_per_partition = 5

    dpEngine = pipeline_dp.DPEngine(pipeline_dp.BudgetAccountant(epsilon=1, delta=1e-10),
    pipeline_dp.LocalPipelineOperations())
    bound_op = dpEngine._bound_cross_partition_contributions(
                          inp_col,
                          max_partitions_contributed=max_partitions_contributed,
                          max_contributions_per_partition=max_contributions_per_partition,
                          aggregator_fn=DPEngineTest.aggregator_fn)

    self.assertEqual(len(bound_op), 4)
    self.assertTrue(all(map(lambda op_val : op_val[1][0] <= max_contributions_per_partition, bound_op)))
    dict_of_pid_to_pk = defaultdict(lambda:[])
    for key, _ in bound_op:
      dict_of_pid_to_pk[key[0]].append(key[1])
    self.assertEqual(len(dict_of_pid_to_pk), 2)
    self.assertTrue(all(map(lambda key : len(dict_of_pid_to_pk[key]) <= max_partitions_contributed, dict_of_pid_to_pk)))

if __name__ == '__main__':
  unittest.main()
