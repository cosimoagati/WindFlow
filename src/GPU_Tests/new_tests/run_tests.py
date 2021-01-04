#!/usr/bin/env python3
from subprocess import run, PIPE
import sys
from statistics import mean

FILE = 'testresults.txt'
STATELESS_TESTS = ['gpu_map_stateless', 'gpu_filter_stateless']
STATEFUL_TESTS = ['gpu_map_stateful', 'gpu_filter_stateful']
EXTRA_MAP_TESTS = ['gpu_map_stateful_one_per_warp',
                   'gpu_map_stateful_no_warps']
EXTRA_FILTER_TESTS = ['gpu_filter_stateful_one_per_warp',
                      'gpu_filter_stateful_no_warps']
KEY_AMOUNTS = [1, 10, 100, 500, 960, 1000, 2000, 4000, 8000, 10000]
SOURCES_NUM = 14
STREAM_LENGTH = 50000000
BATCH_LENGTH = 10000
WORKER_NUM = 1
DATASET_FILE = 'sensors.dat'


def print_usage_and_exit():
    """Print out correct usage in case of wrong invocation."""
    sys.stderr.write('Use as ' +
                     sys.argv[0] +
                     ' all|stateless|stateful|mapextra|filterextra\n')
    sys.exit(-1)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage_and_exit()

    choice = sys.argv[1].lower()
    if choice == 'all':
        tests = (STATELESS_TESTS + STATEFUL_TESTS + EXTRA_MAP_TESTS +
                 EXTRA_FILTER_TESTS)
    elif choice == 'stateless':
        tests = STATELESS_TESTS
    elif choice == 'stateful':
        tests = STATEFUL_TESTS
    elif choice == 'mapextra':
        tests = EXTRA_MAP_TESTS
    elif choice == 'filterextra':
        tests = EXTRA_FILTER_TESTS
    else:
        print_usage_and_exit()

    print('Starting tests, writing results to ' + FILE)
    with open(FILE, 'w') as output_file:
        for test in tests:
            for keynum in KEY_AMOUNTS:
                arglist = ['./' + test, '-s', str(SOURCES_NUM), '-b',
                           str(BATCH_LENGTH), '-n', str(WORKER_NUM), '-k',
                           str(keynum), '-f', DATASET_FILE]
                print(arglist)
                output_file.write(str(arglist) + '\n')
                run_results = []
                for i in range(3):
                    # PIPE needed to capture output on Python < 3.7
                    output = run(arglist, stdout=PIPE, check=True)
                    run_results.append(int(output.stdout))
                output_file.write(str(round(mean(run_results))) + '\n')
                output_file.flush()

            output_file.write('\n\n')
            output_file.flush()
