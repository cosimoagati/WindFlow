#!/usr/bin/env python3
import subprocess

with open("stateful_results.txt", "w") as output_file:
    for testnum in range(1, 7):
        for keynum in [1, 10, 100, 500, 960, 1000, 2000, 4000, 8000, 10000]:
            arglist = ["./test_stateful_v" + str(testnum), "-l 50000000",
                       "-b 10000", "-n 1", "-k " + str(keynum)]
            print(arglist)
            subprocess.run(arglist, stdout=output_file)
