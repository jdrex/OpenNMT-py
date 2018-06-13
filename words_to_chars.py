import sys
import re
import codecs

numberF = open('/data/sls/scratch/jdrexler/OpenNMT-py/number_words_uniq.txt')
numberDict = dict()
for line in numberF:
    p = line.strip().split()
    numberDict[p[0]] = " ".join(p[1:])
inF = codecs.open(sys.argv[1], 'r', encoding="utf-8")
#outF = codecs.open(sys.argv[1] + ".chars", 'w', encoding="utf-8")

lm = False

ngram = 1
if len(sys.argv) >= 3:
    ngram = int(sys.argv[2])
    outF = codecs.open(sys.argv[1] + ".chars.nopunc." + sys.argv[2], 'w', encoding="utf-8")
else:
    outF = codecs.open(sys.argv[1] + ".chars.nopunc", 'w', encoding="utf-8")

longest_line = 0
longLines = 0
for line in inF:
    if lm:
        words = line.strip().split()
    else:
        words = line.strip().split()[1:]
        
    words = [w for w in words if not (w.startswith('<') and w.endswith('>'))]
    updated_words = []
    for w in words:
        if w in numberDict:
            updated_words.append(numberDict[w])
        else:
            updated_words.append(w)
    words = ' '.join(updated_words).split()
    
    for i in range(len(words)):
        if len(words[i]) > 1 and not words[i][0].isalnum():
           words[i] = words[i][1:]
        words[i] = re.sub('\+', '\'', words[i])
        #words[i] = re.sub('[^0-9a-zA-Z.,\'\-]+', '', words[i])
        words[i] = re.sub('[^0-9a-zA-Z]+', '', words[i])
                        
    charStrings = [" ".join(w) for w in words]
    outLine = " SPACE ".join(charStrings)

    #allChars = outLine.split()
    #nGrams = [allChars[i:i+ngram] for i in range(len(allChars) - ngram + 1)]
    #charStrings = ["".join(w) for w in nGrams]
    #outLine = " ".join(charStrings)

    if len(outLine)/2 > 200:
        longLines += 1
    if len(outLine)/2 > longest_line:
        longest_line = len(outLine)/2
    outF.write(outLine + "\n")

print longest_line
print longLines
