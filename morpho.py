import sys, os
out = open("/tmp/words", 'w')
data_dir = sys.argv[1]
morfessor_dir = sys.argv[2]
counts = {}

for f in ["train.txt", "valid.txt", "test.txt"]:
    for l in open(data_dir + "/" + f):
        words = l.strip().split()
        for w in words:
            counts.setdefault(w, 0)
            counts[w] += 1

for k, v in counts.iteritems():
    w = k.replace("\\", "")
    w = k.replace("/", "")
    if w.strip():
        print >>out, v, w

os.system("cd %s; cp /tmp/words mydata; rm mydata.gz; gzip mydata; rm baseline*; make clean; make; gunzip segmentation.final.gz"%(morfessor_dir,))

f = open("%s/segmentation.final"%(morfessor_dir,))
words = {}
for line in f:
    if line[0] == "#": 
        continue
    word_parts = line.replace("+", "").strip().split()[1:]
    words["".join([part.split("/")[0] for part in word_parts])] =  word_parts

morpho = open(data_dir + "/morpho.txt", "w")
for word, factors in words.iteritems():
    print >>morpho, word, " ".join(factors)
print >>morpho, "+", "+"
print >>morpho, "|", "|"
