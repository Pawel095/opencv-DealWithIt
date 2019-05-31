import unittest


class TestaverageQueue(unittest.TestCase):
    def test_averageQueue(self):
        from averageQueue import averageQueue as aq;
        avg = aq(3)
        aq.add(3)
        aq.add(None)
        aq.add(3)
        self.assertAlmostEqual(avg.get_avg(), 2)

        aq.add(500)
        self.assertAlmostEqual(aq.get_avg(), 168.67, 2)
