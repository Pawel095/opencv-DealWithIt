import unittest


class TestMath(unittest.TestCase):
    def test_get_dist(self):
        from mathUtils import get_dist
        e=[[0,-1],[0,4]]
        self.assertAlmostEqual(get_dist(e), 5)

    def test_approach(self):
        from mathUtils import approach
        val=0
        desired=5
        step=0.5
        val=approach(val,desired,step)
        self.assertAlmostEqual(val,2.5)
        val = approach(val, desired, step)
        self.assertAlmostEqual(val,3.75)
