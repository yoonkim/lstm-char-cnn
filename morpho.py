import sys, os
out = open("/tmp/words", 'w')
for l in sys.stdin:
    words = l.strip().split()
    for w in words:
        print >>out,  w

os.system("perl morfessor1.0.pl -data /tmp/words > /tmp/morph")
f = open("/tmp/morph")
words = {}
for line in f:
    if line[0] == "#": continue
    word_parts = line.replace("+", "").strip().split()[1:]
    words["".join(word_parts)] =  word_parts

for word, factors in words.iteritems():
    print word, " ".join(factors)
