import unittest

from utility_analysis_new import histogram_report


class HistogramReportTest(unittest.TestCase):

    def test_calc_partition_contrib_freq(self):
        hr = histogram_report.HistogramReport([(101, 2), (102, 1), (103, 3),
                                               (104, 1), (105, 2)])
        report = hr.calc_partition_contrib_freq()
        self.assertEqual(report[1], 2)
        self.assertEqual(report[2], 2)
        self.assertEqual(report[3], 1)


if __name__ == '__main__':
    unittest.main()
