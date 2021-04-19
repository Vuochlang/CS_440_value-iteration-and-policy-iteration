# reference: https://github.com/python/cpython/blob/master/Lib/unittest/runner.py


import sys
import time
import json
import warnings

import unittest
from unittest import result
from unittest.signals import registerResult
from unittest import TextTestResult

class WSUVTextTestRunner(unittest.TextTestRunner):
    resultclass = TextTestResult

    def __init__(self, **kwargs):
        """Construct a TextTestRunner.

        Subclasses should accept **kwargs to ensure compatibility as the
        interface changes.
        """
        super(WSUVTextTestRunner, self).__init__(**kwargs)


    def run(self, test):
        "Run the given test case or test suite."

        # Load the json file first, so a student can't overwrite it...
        with open('wsuvtest.json') as fin:
            config = json.load(fin)

        result = super(WSUVTextTestRunner, self).run(test)
        run = result.testsRun
        failed, errored = 0,0
        if not result.wasSuccessful():
            failed, errored = len(result.failures), len(result.errors)
        pts = (run-failed-errored)*1.0*config['scores']['Correctness']/run

        outstr = ["Total points %.1f out of %.1f"%(pts, config['scores']['Correctness']) ,
                  '{"scores": {"Correctness": %.1f}}'%(pts)]
        return "\n".join(outstr)

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true')
    args = parser.parse_args()
    
    test_modules = unittest.defaultTestLoader.discover(start_dir='.',
                                                       pattern='*tests.py',
                                                       top_level_dir=None)

    try:
        fout = sys.stdout
        if args.f:
            fout = open('wsuvpyunitrunner.out', 'wt')
            
        runner = WSUVTextTestRunner(verbosity=5, stream=fout)
        result = runner.run(test_modules)
        print(result, file=fout)
    finally:
        if args.f and fout:
            fout.close()
