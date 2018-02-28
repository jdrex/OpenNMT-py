import sys
import re

inF = open(sys.argv[1], 'r')
outF = open(sys.argv[1] + ".chars", 'w')

for line in inF:
    words = line.strip().split()[1:]
    words = [w for w in words if w != "<NOISE>"]
    for i in range(len(words)):
        if len(words[i]) > 1 and not words[i][0].isalnum():
            words[i] = words[i][1:]
        words[i] = re.sub('[^0-9a-zA-Z.,\']+', '', words[i])
                        
    charStrings = [" ".join(w) for w in words]
    outLine = " SPACE ".join(charStrings)
    outF.write(outLine + "\n")
