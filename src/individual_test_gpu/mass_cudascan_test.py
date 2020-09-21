#!/usr/bin/env python3

import random
import subprocess

arglist = []
for i in range(64):
    arglist.append(str(random.randint(0, 1)))
    print(arglist, flush=True)
    subprocess.run(["./a.out"] + arglist)
