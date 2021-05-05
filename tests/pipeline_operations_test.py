import unittest

from pipeline_dp.pipeline_operations import LocalPipelineOperations


class PipelineOperationsTest(unittest.TestCase):
  pass

class LocalPipelineOperationsTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.ops = LocalPipelineOperations()

  def test_local_map(self):
    some_map = self.ops.map([1,2,3], lambda x: x)
    # some_map is its own consumable iterator
    self.assertIs(some_map, iter(some_map))

    self.assertEqual(list(self.ops.map([1,2,3], str)),
                     ["1", "2", "3"])
    self.assertEqual(list(self.ops.map(range(5), lambda x: x ** 2)),
                     [0, 1, 4, 9, 16])

if __name__ == '__main__':
  unittest.main()
