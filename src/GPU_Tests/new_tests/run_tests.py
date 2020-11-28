#!/usr/bin/env python3
import subprocess
import sys

FILE = 'testresults.txt'
STATELESS_TESTS = ['gpu_map_stateless', 'gpu_filter_stateless']
STATEFUL_TESTS = ['gpu_map_stateful', 'gpu_filter_stateful']


def error():
    sys.stderr.write('Use as ' + sys.argv[0] + ' all|stateless|stateful\n')
    sys.exit(-1)


if len(sys.argv) != 2:
    error()

choice = sys.argv[1].lower()
if choice == 'all':
    tests = STATELESS_TESTS + STATEFUL_TESTS
elif choice == 'stateless':
    tests = STATELESS_TESTS
elif choice == 'stateful':
    tests = STATEFUL_TESTS
else:
    error()

print('Starting tests, writing results to ' + FILE)
with open(FILE, 'w') as output_file:
    for test in tests:
        for keynum in [1, 10, 100, 500, 960, 1000, 2000, 4000, 8000, 10000]:
            arglist = ['./' + test, '-s', '1', '-b', '10000', '-n', '1', '-k',
                       str(keynum), '-f', 'sensors.dat']
            print(arglist)
            output_file.write(str(arglist) + '\n')
            output_file.flush()

            subprocess.run(arglist, stdout=output_file, check=True)
            output_file.write('\n')
            output_file.flush()

        output_file.write('\n\n')
        output_file.flush()
