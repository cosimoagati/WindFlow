#!/usr/bin/env python3
import sys
from itertools import product
from socket import gethostname
from statistics import mean
from subprocess import run, PIPE

FILE = 'testresults.txt'
STATELESS_TESTS = ['gpu_map_stateless', 'gpu_filter_stateless']
EXTRA_MAP_STATELESS_TESTS = ['gpu_map_stateless_one_per_warp'
                             'gpu_map_stateless_no_warps']
EXTRA_FILTER_STATELESS_TESTS = ['gpu_filter_stateless_one_per_warp'
                                'gpu_filter_stateless_no_warps']
STATEFUL_TESTS = ['gpu_map_stateful', 'gpu_filter_stateful']
EXTRA_MAP_TESTS = ['gpu_map_stateful_one_per_warp',
                   'gpu_map_stateful_no_warps']
EXTRA_FILTER_TESTS = ['gpu_filter_stateful_one_per_warp',
                      'gpu_filter_stateful_no_warps']
KEY_AMOUNTS = [1, 10, 100, 500, 960, 1000, 2000, 4000, 8000, 10000]
SOURCES_NUMS = ([1, 2, 4, 6, 8, 10, 12, 14] if gethostname() == 'pianosa' else
                [1, 2])
# STREAM_LENGTH = 50000000
BATCH_LENGTHS = [1000, 5000, 10000]
RUNS_NUM = 1
WORKER_NUM = 1
DATASET_FILE = 'sensors.dat'


def print_usage_and_exit():
    """Print out correct usage in case of wrong invocation."""
    sys.stderr.write('Use as ' + sys.argv[0] + ' all|testnames...\n')
    sys.exit(-1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage_and_exit()

    tests = []
    for test in [arg.lower() for arg in sys.argv[1:]]:
        if test == 'all':
            tests = (STATELESS_TESTS + EXTRA_MAP_STATELESS_TESTS +
                     EXTRA_FILTER_STATELESS_TESTS + STATEFUL_TESTS +
                     EXTRA_MAP_TESTS + EXTRA_FILTER_TESTS)
        else:
            tests.append(test)

    print('Starting tests, writing results to ' + FILE)
    with open(FILE, 'w') as output_file:
        params = product(tests, BATCH_LENGTHS, SOURCES_NUMS, KEY_AMOUNTS)
        for test, batch_length, sources, keynum in params:
            arglist = ['./' + test, '-s', str(sources), '-b',
                       str(batch_length), '-n', str(WORKER_NUM),
                       '-k', str(keynum), '-f', DATASET_FILE]
            print(arglist)
            output_file.write(str(arglist) + '\n')
            run_results = []
            for i in range(RUNS_NUM):
                # PIPE needed to capture output on Python < 3.7
                output = run(arglist, stdout=PIPE, check=True)
                run_results.append(float(output.stdout))
            output_file.write(str(round(mean(run_results))) + '\n')
            output_file.flush()
