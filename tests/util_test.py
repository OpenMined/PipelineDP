from absl.testing import absltest
from absl.testing import parameterized

import pipeline_dp.util as util


class UtilTest(parameterized.TestCase):

    def _check_is_subset(self, a: list, b: list) -> bool:
        return set(a).issubset(set(b))

    @parameterized.parameters({
        "size": 1,
        "sample_sizes": [1, 2, 10, 20]
    }, {
        "size": 10,
        "sample_sizes": [1, 2, 10, 20]
    })
    def test_choose_from_list_ints(self, size, sample_sizes):
        a = list(range(size))
        for sample_size in sample_sizes:
            sample = util.choose_from_list_without_replacement(a, sample_size)
            self.assertTrue(self._check_is_subset(sample, a))

    @parameterized.parameters({
        "size": 1,
        "sample_sizes": [1, 2, 10, 20]
    }, {
        "size": 10,
        "sample_sizes": [1, 2, 10, 20]
    })
    def test_choose_from_list_tuples(self, size, sample_sizes):
        a = [(i, i) for i in range(size)]
        for sample_size in sample_sizes:
            sample = util.choose_from_list_without_replacement(a, sample_size)
            self.assertTrue(self._check_is_subset(sample, a))


if __name__ == '__main__':
    absltest.main()
