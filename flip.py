import sys
import re

inf = open(sys.argv[1], 'r')
outf = open(sys.argv[1] + '.flip', 'w')

for line in inf:
    line = line.strip()
    line = re.sub('N', '$', line)
    line = re.sub('O', 'N', line)
    line = re.sub('\$', 'O', line)
    outf.write(line + '\n')

inf.close()
outf.close()
