import sys
import re

inF = open(sys.argv[1], 'r')
outF = open(sys.argv[1] + ".words", 'w')

for line in inF:
    chars = line.strip().split()
    for i in range(len(chars)):
        if chars[i] == "SPACE":
            chars[i] = " "
    outLine = "".join(chars)
    outF.write(outLine + "\n")
