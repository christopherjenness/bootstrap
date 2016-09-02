import unittest
from bootstrap import bootstrap
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
        self.matrix_data = np.random.normal(100, 10, size=(100,100))
    

class ResamplingTestCase(BootstrapInit):
    def testNonparametric(self):
        bootstrap_data = bootstrap.bootstrap_sample(self.normal_data)
        self.assertAlmostEqual(np.mean(self.normal_data)/10000, np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.normal_data))
        
    def testNormalParametric(self):
        bootstrap_data = bootstrap.bootstrap_sample(self.normal_data, parametric='normal')
        self.assertAlmostEqual(np.mean(self.normal_data)/10000, np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.normal_data))
        
    def testUniformParametric(self):
        bootstrap_data = bootstrap.bootstrap_sample(self.uniform_data, parametric='uniform')
        self.assertAlmostEqual(np.mean(self.uniform_data)/10000, np.mean(bootstrap_data)/10000, 3)
        self.assertEqual(len(bootstrap_data), len(self.uniform_data))
        
class TwoSampleTestCase(BootstrapInit):
    def testMeanDifference(self):
        mean_difference = bootstrap.compare_means(self.normal_data2, self.normal_data)
        self.assertAlmostEqual(mean_difference/100000, 0.001, 3)
        
    def testTStatisticBig(self):
        t_statistic = bootstrap.t_test_statistic(self.normal_data2, self.normal_data)
        self.assertAlmostEqual(t_statistic / 10000, 0.007, 3)
        
    def testTStatisticSmall(self):
        t_statistic = bootstrap.t_test_statistic(self.normal_data, self.normal_data)
        self.assertEqual(t_statistic, 0)

if __name__ == '__main__':
    unittest.main()
    
