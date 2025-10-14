import unittest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import sys

# Mock moviepy editor at the top level
mock_moviepy_editor = MagicMock()
sys.modules['moviepy.editor'] = mock_moviepy_editor

from preprocessing import seg

class TestPreprocessing(unittest.TestCase):

    def test_seg_adaptive_thresholding(self):
        # Create a dummy image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Add a white box to simulate text
        img[40:60, 20:80] = [255, 255, 255]

        mask = seg(img)

        # Check that the mask is a 3-channel image of the same size
        self.assertEqual(mask.shape, img.shape)
        # Check that the mask is binary (contains only 0 and 255)
        self.assertTrue(np.all(np.logical_or(mask == 0, mask == 255)))

if __name__ == '__main__':
    unittest.main()