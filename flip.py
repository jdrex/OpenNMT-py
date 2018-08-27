import sys
import re

inf = open(sys.argv[1], 'r')
outf = open(sys.argv[1] + '.flip', 'w')

for line in inf:
    line = line.strip()
    line = re.sub('M', '$', line)
    line = re.sub('U', 'M', line)
    line = re.sub('\$', 'U', line)
    outf.write(line + '\n')

inf.close()
outf.close()
