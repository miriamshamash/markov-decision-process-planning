import unittest
from MDP import MDP

class TestPolicyExtract(unittest.TestCase):
    def test_policy_linear_1(self):
        linear = MDP("./maps/linear.json", gamma=0.5)
        u = linear.value_iteration()
        policy = linear.policy_extraction(u)
        target = [[None, 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'RIGHT', 'RIGHT', None]]

        self.assertEqual(policy, target)

    def test_policy_linear_2(self):
        linear = MDP("./maps/linear.json", gamma=0.8)
        u = linear.value_iteration()
        policy = linear.policy_extraction(u)
        target = [[None, 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', 'LEFT', None]]

        self.assertEqual(policy, target)

    def test_policy_uturn(self):
        linear = MDP("./maps/uturn.json", gamma=0.8)
        u = linear.value_iteration()
        policy = linear.policy_extraction(u)
        target = [['RIGHT', 'RIGHT', 'DOWN', 'LEFT', 'LEFT'], 
                  ['UP',     None,   'DOWN',  None, 'UP'], 
                  ['UP',     None,   'DOWN',  None, 'UP'], 
                  ['UP',     None,    None,   None, 'UP']]

        self.assertEqual(policy, target)

    def test_policy_center_1(self):
        linear = MDP("./maps/center.json", gamma=0.1)
        u = linear.value_iteration()
        policy = linear.policy_extraction(u)
        target = [[None,    'LEFT',  'DOWN', 'RIGHT', None], 
                  ['UP',     None,  'DOWN',   None, 'UP'], 
                  ['RIGHT', 'RIGHT', None,   'LEFT', 'LEFT'], 
                  ['DOWN',   None,  'UP',     None, 'DOWN'], 
                  [None,    'LEFT', 'UP',    'RIGHT', None]]

        self.assertEqual(policy, target)

    def test_policy_center_2(self):
        linear = MDP("./maps/center.json", gamma=0.5)
        u = linear.value_iteration()
        policy = linear.policy_extraction(u)
        target = [[None,    'RIGHT', 'DOWN', 'LEFT', None], 
                  ['DOWN',   None,  'DOWN',  None,  'DOWN'], 
                  ['RIGHT', 'RIGHT', None,  'LEFT', 'LEFT'], 
                  ['UP',     None,   'UP',  None,  'UP'], 
                  [None,    'RIGHT', 'UP',  'LEFT', None]]

        self.assertEqual(policy, target)

