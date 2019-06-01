import unittest


class TestaverageQueue(unittest.TestCase):
    def test_averageQueue(self):
        from averageQueue import averageQueue as aq;
        avg = aq(3)
        avg.add(3)
        avg.add(None)
        avg.add(3)
        self.assertAlmostEqual(avg.get_avg(), 3)

        avg=aq(3)
        avg.add(3)
        avg.add(3)
        avg.add(3)
        avg.add(500)
        self.assertAlmostEqual(avg.get_avg(), 168.67, 2)
