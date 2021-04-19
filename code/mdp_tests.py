import re
import unittest
from mdp import *
import subprocess
import timeout_decorator
import time

class MDPTestCase(unittest.TestCase):

    def vi_initialization_check(self):
        rfn = [None, -0.04, -0.04, 1, -1].__getitem__
        d = value_iteration(TwoXTwoMDP, 1.0, rfn, True, n=1)

        if (abs(d[1] - -0.08) < .001):
            # looks like U[s] was initialized to rfn(s)
            initialization = 'reward'
        elif (abs(d[1] - -0.04) < .001):
            # looks like U[s] was initialized to 0
            initialization = 'zero'
        else:
            raise ValueError("After 1 iteration, utilities on state 1 should be either -0.08 or -0.04....depending on your initialization method")

        return initialization


    def test_initializaion(self):
        self.assertIsNotNone( self.vi_initialization_check() )

        
    def test_vi_2x2_quiescence(self):
        init = self.vi_initialization_check()
        rfn = [None, -0.04, -0.04, 1, -1].__getitem__
        d = value_iteration(TwoXTwoMDP, 1.0, rfn, True)
        
        self.assertAlmostEqual(d[1], 0.660, 3, 'After 1 iteration, expected ~0.660 in state 1 of 2x2 environment')
        self.assertAlmostEqual(d[2], 0.918, 3, 'After completion, expected ~0.918 in state 2 of 2x2 environment')



    def test_vi_4x3_quiescence(self):
        init = self.vi_initialization_check()
        rfn = lambda s: {(4,2):-1, (4,3):1}.get(s, -0.04)
        d = value_iteration(FourXThreeMDP, 1.0, rfn, True)

        U = { (1,1): .71, (1,2): .76, (1,3): .81, 
              (2,1): .66, (2,3): .87,
              (3,1): .61, (3,3): .92 }
        # The slides show (3,2) converging to 0.67, but hand calculation suggests that may be
        # wrong and the correct value may be 0.66, so I've left that off the test for now...
        
        for state in U:
            self.assertAlmostEqual(d[state], U[state], 2,
                                   'Expected utilities in state %s to match the slides...'%(str(state)))
    

        

    def test_pi_2x2_quiescence(self):
        rfn = [None, -0.04, -0.04, 1, -1].__getitem__
        pi = {s:'L' for s in TwoXTwoMDP['stategraph']}
        pstar = {1: 'U', 2: 'R'}
        p = policy_iteration(TwoXTwoMDP, 1.0, rfn, pi, True)

        for state in pstar:
            self.assertEqual(pstar[state], p[state], "Expected policy on state %s to match slides..."%(str(state)))


    def test_pi_4x3_quiescence(self):
        rfn = lambda s: {(4,2):-1, (4,3):1}.get(s, -0.04)
        pi = {s:'L' for s in FourXThreeMDP['stategraph']}
        pstar = { (1,1): 'U', (1,2): 'U', (1,3): 'R',
              (2,1): 'L', (2,3): 'R',
              (3,3): 'R' }

        p = policy_iteration(FourXThreeMDP, 1.0, rfn, pi, True)

        for state in pstar:
            self.assertEqual(pstar[state], p[state], "Expected policy on state %s to match slides..."%(str(state)))
            
        
                


