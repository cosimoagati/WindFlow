#!/usr/bin/env python3
import subprocess

FILE = "stateful-results-uniform.txt"
KEY_AMOUNTS = [1, 10, 100, 500, 960, 1000, 2000, 4000, 8000, 10000]
STREAM_LENGTH = 50000000
BATCH_LENGTH = 10000
WORKER_NUM = 1

if __name__ == '__main__':
    print('Starting tests, writing results to ' + FILE)
    with open(FILE, 'w') as output_file:
        for testnum in range(1, 7):
            for keynum in KEY_AMOUNTS:
                arglist = ['./test_stateful_v' + str(testnum), '-l',
                           str(STREAM_LENGTH), '-b', str(BATCH_LENGTH), '-n',
                           str(WORKER_NUM), '-k', str(keynum)]
                print(arglist)
                output_file.write(str(arglist) + '\n')
                output_file.flush()

                subprocess.run(arglist, stdout=output_file, check=True)
                output_file.write('\n')
                output_file.flush()

            output_file.write('\n\n')
            output_file.flush()

    with open(FILE, 'r+') as output_file:
        lines = output_file.readlines()
        output_file.seek(0)
        for line in lines:
            if not line.startswith('[SINK]') or 'total' in line:
                output_file.write(line)
        output_file.truncate()
