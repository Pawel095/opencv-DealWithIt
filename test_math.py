import unittest


class TestMath(unittest.TestCase):
    def test_get_dist(self):
        from mathUtils import get_dist
        e=[[0,-1],[0,4]]
        self.assertAlmostEqual(get_dist(e), 5)
