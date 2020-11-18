#!/usr/bin/env python3
import subprocess

FILE = 'testresults.txt'

print('Starting tests, writing results to ' + FILE)

with open(FILE, 'w') as output_file:
    for test in ['gpu_map_stateless', 'gpu_map_stateful',
                 'gpu_filter_stateless', 'gpu_filter_stateful']:
        for keynum in [1, 10, 100, 500, 960, 1000, 2000, 4000, 8000, 10000]:
            arglist = ['./' + test, '-s', '1',
                       '-b', '10000', '-n', '1', '-k', str(keynum),
                       '-f', 'sensors.dat']
            print(arglist)
            output_file.write(str(arglist) + '\n')
            output_file.flush()

            subprocess.run(arglist, stdout=output_file, check=True)
            output_file.write('\n')
            output_file.flush()

        output_file.write('\n\n')
        output_file.flush()

# with open(FILE, 'r+') as output_file:
#     lines = output_file.readlines()
#     output_file.seek(0)
#     for line in lines:
#         if not line.startswith('[SINK]') or 'total' in line:
#             output_file.write(line)
#     output_file.truncate()
