import unittest
from Learner.GreedyLearner import *


class MyTestCase(unittest.TestCase):
    def test_greedyLearnerFirstOptima(self):
        gLearner = GreedyLearner()
        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 0, 0])
        gLearner.update(100)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [1, 0, 0, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 1, 0, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 1, 0, 0])
        gLearner.update(10)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 1, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 0, 1])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 0, 0])


    def test_greedyLearnerSecondOptima(self):
        gLearner = GreedyLearner()
        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [1, 0, 0, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 1, 0, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 1, 0, 0])
        gLearner.update(10)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 1, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 0, 0, 1])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [1, 0, 1, 0, 0])
        gLearner.update(1)

        self.assertEqual(gLearner.optimal_configuration_margin, 10)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 1, 1, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 2, 0, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 1, 1, 0])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 1, 0, 1])
        gLearner.update(1)

        pulledArm = gLearner.pull_arm()
        self.assertEqual(pulledArm, [0, 0, 1, 0, 0])
        gLearner.update(1)

        for i in range(0,1000):
            pulledArm = gLearner.pull_arm()
            self.assertEqual(pulledArm, [0, 0, 1, 0, 0])
            gLearner.update(1000)

        self.assertEqual(True, True)

    def test_greedyLearnerLaterOptima(self):
        gLearner = GreedyLearner()
        for i in range(0,300):
            pulledArm = gLearner.pull_arm()
            #print(pulledArm)
            gLearner.update(i)

        for i in range(0,298):
            pulledArm = gLearner.pull_arm()
            #print(pulledArm)
            gLearner.update(1000)

        self.assertEqual(pulledArm, [3, 3, 3, 3, 3])

if __name__ == '__main__':
    unittest.main()
