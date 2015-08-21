import sys, os
out = open("/tmp/words", 'w')

data_dir = sys.argv[1]
morfessor_dir = sys.argv[2]


for f in ["train.txt", "valid.txt", "test.txt"]:
    for l in open(data_dir + "/" + f):
        words = l.strip().split()
        for w in words:
            print >>out, 1,  w.replace("\\", "")

os.system("cd %s/train; cp /tmp/words mydata; rm mydata.gz; gzip mydata; make clean; make; cp segmentation.final.gz /tmp/morph.gz; rm /tmp/morph; gunzip /tmp/morph.gz"%(morfessor_dir,))

f = open("/tmp/morph")
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
