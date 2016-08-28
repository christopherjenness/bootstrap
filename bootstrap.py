"""
Library for boostraping statistics

Citation:
Efron, Bradley, and Robert J. Tibshirani.
An introduction to the bootstrap. CRC press, 1994.
"""

import numpy as np
from scipy.stats import sem

class Bootstrap:
    
    def __init__(self):
        return
    
    
    def resample(self, data, parametric=False):
        """
        Resamples data by random sample with replacement
        
        Args:
            data (np.array): array of data to resample
            parametric (str) in ['gaussian', 'uniform']: parametric distribution to resample from
                if False, use nonparametric bootstrap
        
        Returns:
            resampled data (array)
        """
        if parametric == 'gaussian':
            # TODO code for gaussian resampling
            pass
        elif parametric == 'uniform':
            # TODO code for uniform resampling
            pass
        else:
            sample_size = len(data)
            indices = [np.random.randint(0, sample_size) for i in range(sample_size)]
            return data[indices]
        
    def bootstrap_statistic(self, data, func=np.mean, n_samples=50, parametric=False):
        """
        Bootstraps a statistic and calculates the standard error of the statistic
        
        Args:
            data (array): array of data to calculate statistic and SE of statistic
            func (function): statistical function to calculate on data
                examples: np.mean, np.median
            n_samples (int): number of bootstrap samples to 
            
        Returns:
            statistic (tuple), sem (tuple)
            Returns the bootstrapped statistic and the SEM of the statistic
        """
        statistics = []
        for sample in range(n_samples):
            resample = self.resample(data, parametric=parametric)
            statistic = func(resample)
            statistics.append(statistic)
        return (np.mean(statistics), sem(statistics))
        

data = np.random.uniform(0, 1, 100)
bootstrap = Bootstrap()
bootstrap_mean = bootstrap.bootstrap_statistic(data)
print(bootstrap_mean)



























