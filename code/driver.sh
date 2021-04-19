#!/bin/bash


# driver.sh - The simplest autograder using JUnitTests.
#   Usage: ./driver.sh

# Compile the code

echo "$Cleaning..."
make --no-print-directory clean

# Run the code
echo "Running..."
make --no-print-directory run
status=$?
if [ ${status} -ne 0 ]; then
    echo "Failure: testsuite failed with nonzero exit status of ${status}"
    echo "{\"scores\": {\"Correctness\": 0}}"
fi

if [ -e wsuvpyunitrunner.out ]
  then cat wsuvpyunitrunner.out
fi

exit
