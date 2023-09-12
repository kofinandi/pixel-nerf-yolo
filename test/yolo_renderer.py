import unittest
from src.render import yolo
import sys
import os
import torch

class MyTestCase(unittest.TestCase):
    def test_sample_coarse(self):
        renderer = yolo.YoloRenderer(128, 1024)
        ray_batch = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 10.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.1, 10.0],
            ]
        )
        print(renderer.sample_coarse(ray_batch))
        print(renderer.sample_coarse(ray_batch).shape)

    def test_forward(self):
        renderer = yolo.YoloRenderer(128, 1024)
        ray_batch = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 10.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.1, 10.0],
            ]
        )
        renderer(ray_batch)


if __name__ == '__main__':
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    )

    unittest.main()
