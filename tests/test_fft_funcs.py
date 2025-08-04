import numpy as np
from uncertainty_estimation import fft_funcs
import unittest
import sys

sys.path.append("../") 

from interpolation_uncertainty.uncertainty_estimation import fft_funcs

class Test_FFT_Functions(unittest.TestCase): 
    def test_get_column_indices(self):
        indices = fft_funcs.get_column_indices(10, 1, 1, 1)
        self.assertGreater(len(indices), 1)
        self.assertIsInstance(indices[0], int)
        
        




if __name__ == '__main__':
    unittest.main()