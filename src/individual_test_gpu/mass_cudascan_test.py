#!/usr/bin/env python3

import random
import subprocess
import sys

if (len(sys.argv) == 1):
    number_of_runs = 64
else:
    number_of_runs = int(sys.argv[1])

arglist = []
for i in range(number_of_runs):
    arglist.append(str(random.randint(0, 1)))
    print(arglist, flush=True)
    subprocess.run(["./a.out"] + arglist)
