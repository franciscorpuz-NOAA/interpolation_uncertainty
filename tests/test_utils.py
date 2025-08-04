import numpy as np
import unittest
import sys
from pathlib import Path

sys.path.append("../") 

from interpolation_uncertainty.uncertainty_estimation import tiffread_utils

class Test_Utils_Functions(unittest.TestCase):
    def test_loadFile(self):
        test_surface = tiffread_utils.load_file('sample_tiff.tif', 'tests')
        self.assertIsInstance(test_surface, dict)
        self.assertGreater(len(test_surface['depth'].shape), 1)
        self.assertGreater(test_surface['resolution'], 1)
    
    def test_remove_edge_nans(self):
        test_surface = tiffread_utils.load_file('sample_tiff.tif', 'tests')
        raw_array = test_surface['depth']
        ndv_val = test_surface['ndv']
        new_surface = tiffread_utils.remove_edge_Nans(raw_array, ndv=ndv_val)
        self.assertLessEqual(new_surface.size, raw_array.size)
        self.assertTrue(np.any(new_surface))



if __name__ == '__main__':
    unittest.main()