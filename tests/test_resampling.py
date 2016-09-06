import unittest
from bootstrap.bootstrap import bootstrap_sample, jackknife_sample, \
        compare_means, t_test_statistic, two_sample_testing, \
        bootstrap_matrixsample, bootstrap_statistic, jackknife_statistic
import numpy as np


class BootstrapInit(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.normal_data = np.random.normal(100, 10, size=100)
        np.random.seed(0)
        self.normal_data2 = np.random.normal(200, 10, size=100)
        np.random.seed(0)
        self.uniform_data = np.random.uniform(0, 100, size=100)
        np.random.seed(0)
        self.poisson_data = np.random.poisson(10, size=(100))        
        np.random.seed(0)
        self.matrix_data = np.random.normal(100, 10, size=(100, 100))

class ResamplingTestCase(BootstrapInit):
    def testNonparametric(self):
        bootstrap_data = bootstrap_sample(self.normal_data)
        self.assertAlmostEqual(np.mean(self.normal_data)/10000,
                               np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.normal_data))

    def testNormalParametric(self):
        bootstrap_data = bootstrap_sample(self.normal_data,
                                          parametric='normal')
        self.assertAlmostEqual(np.mean(self.normal_data)/10000,
                               np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.normal_data))

    def testUniformParametric(self):
        bootstrap_data = bootstrap_sample(self.uniform_data,
                                          parametric='uniform')
        self.assertAlmostEqual(np.mean(self.uniform_data)/10000,
                               np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.uniform_data))

    def testPoissonParametric(self):
        bootstrap_data = bootstrap_sample(self.poisson_data,
                                          parametric='poisson')
        self.assertAlmostEqual(np.mean(self.poisson_data)/10000,
                               np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.poisson_data))

    def testJackknifeSample(self):
        jackknife_data = jackknife_sample(self.uniform_data, 10)
        self.assertAlmostEqual(np.mean(self.uniform_data)/10000,
                               np.mean(jackknife_data)/10000, 3)
        self.assertEqual(len(jackknife_data) + 1, len(self.uniform_data))


class TwoSampleTestCase(BootstrapInit):
    def testMeanDifference(self):
        mean_difference = compare_means(self.normal_data2,
                                        self.normal_data)
        self.assertAlmostEqual(mean_difference/100000, 0.001, 3)

    def testTStatisticBig(self):
        t_statistic = t_test_statistic(self.normal_data2,
                                       self.normal_data)
        self.assertAlmostEqual(t_statistic / 10000, 0.007, 3)

    def testTStatisticZero(self):
        t_statistic = t_test_statistic(self.normal_data,
                                       self.normal_data)
        self.assertEqual(t_statistic, 0)

    def testTwoSampleZero(self):
        ASL = two_sample_testing(self.normal_data2, self.normal_data)
        self.assertEqual(ASL, 0)

    def testTwoSampleBig(self):
        ASL = two_sample_testing(self.normal_data, self.normal_data)
        self.assertGreater(ASL, 0.05)

    def testTwoSampleTTest(self):
        ASL = two_sample_testing(self.normal_data2, self.normal_data,
                                 statistic_func=t_test_statistic)
        self.assertEqual(ASL, 0)

    def testTwoSampleTTestBit(self):
        ASL = two_sample_testing(self.normal_data, self.normal_data,
                                 statistic_func=t_test_statistic)
        self.assertGreater(ASL, 0.05)


class MatrixTestCase(BootstrapInit):

    def testMatrixResamplingCol(self):
        matrix_sample = np.matrix(bootstrap_matrixsample(self.matrix_data,
                                                         axis=1))
        self.assertEqual(np.shape(matrix_sample), np.shape(self.matrix_data))
        self.assertAlmostEqual(np.average(matrix_sample)/100000,
                               np.mean(self.matrix_data)/100000, 3)


class StatisticsTestCast(BootstrapInit):
    def testBootstrapMean(self):
        statistics, statistic, bias, sem, confidence_interval = \
            bootstrap_statistic(self.normal_data)
        self.assertAlmostEqual(statistic/100000, 100/100000, 2)
        self.assertAlmostEqual(np.abs(bias/100), 0.1/100, 3)
        self.assertAlmostEqual(sem/100, 0.1/100, 3)
        self.assertEqual(len(confidence_interval), 2)
        self.assertTrue(confidence_interval[0] < confidence_interval[1])
        self.assertTrue(confidence_interval[1] - confidence_interval[0] < 5)

    def testBoostrapMedian(self):
        statistics, statistic, bias, sem, confidence_interval = \
            bootstrap_statistic(self.normal_data, func=np.median)
        self.assertAlmostEqual(statistic/100000, 100/100000, 2)
        self.assertAlmostEqual(np.abs(bias/100), 0.3/100, 3)
        self.assertAlmostEqual(sem/100, 0.1/100, 3)
        self.assertEqual(len(confidence_interval), 2)
        self.assertTrue(confidence_interval[0] < confidence_interval[1])
        self.assertTrue(confidence_interval[1] - confidence_interval[0] < 5)

    def testBootstrapBCa(self):
        statistics, statistic, bias, sem, confidence_interval = \
            bootstrap_statistic(self.normal_data, bca=True)
        self.assertAlmostEqual(statistic/100000, 100/100000, 2)
        self.assertAlmostEqual(np.abs(bias/100), 0.1/100, 3)
        self.assertAlmostEqual(sem/100, 0.1/100, 3)
        self.assertEqual(len(confidence_interval), 2)
        self.assertTrue(confidence_interval[0] < confidence_interval[1])
        self.assertTrue(confidence_interval[1] - confidence_interval[0] < 5)

    def testBoostrapParams(self):
        statistics, statistic, bias, sem, confidence_interval = \
            bootstrap_statistic(self.normal_data, parametric='normal',
                                bias_correction=True, alpha=0.1)
        self.assertAlmostEqual(statistic/100000, 100/100000, 2)
        self.assertAlmostEqual(np.abs(bias/100), 0.1/100, 3)
        self.assertAlmostEqual(sem/100, 0.2/100, 3)
        self.assertEqual(len(confidence_interval), 2)
        self.assertTrue(confidence_interval[0] < confidence_interval[1])
        self.assertTrue(confidence_interval[1] - confidence_interval[0] < 5)

    def testBootstrapMatrix(self):
        statistics, statistic, bias, sem, confidence_interval = \
            bootstrap_statistic(self.normal_data)
        self.assertAlmostEqual(statistic/100000, 100/100000, 2)
        self.assertAlmostEqual(np.abs(bias/100), 0.1/100, 3)
        self.assertAlmostEqual(sem/100, 0.1/100, 3)
        self.assertEqual(len(confidence_interval), 2)
        self.assertTrue(confidence_interval[0] < confidence_interval[1])
        self.assertTrue(confidence_interval[1] - confidence_interval[0] < 5)

    def testJackknifeMean(self):
        statistic, sem, statistics = jackknife_statistic(self.normal_data)
        self.assertAlmostEqual(statistic/100000,
                               np.mean(self.normal_data)/100000, 3)
        self.assertAlmostEqual(sem/10, 0.1/100, 3)
        self.assertEqual(len(statistics), len(self.normal_data))

if __name__ == '__main__':
    unittest.main()
