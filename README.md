## Neural Language Modeling with Characters
A neural language model (NLM) built on character inputs only. 
The model employs a convolutional neural network (CNN) over character
embeddings to use as inputs into an long short-term memory (LSTM)
recurrent neural network language model (RNNLM).

Note: Code is messy/experimental. Cleaner (and faster) code coming. Paper 
will be posted on arXiv soon

Most of the code is from Andrej Karpathy's character RNN implementation,
available at https://github.com/karpathy/char-rnn

### Requirements
Code is written in Lua and requires Torch. It additionally requires
the `nngraph` and `optim` packages, which can be installed via:
```
luarocks install nngraph
luarocks install optim
```
GPU usage will additionally require:
```
luarocks install cutorch
luarocks install cunn
```

### Data
Data should be put into the `data/` directory, split into train.txt,
valid.txt, and test.txt

Each line of the .txt file should be a sentence. The English Penn 
Treebank data (Tomas Mikolov's pre-processed version with 10K vocab,
widely used by the language modeling community) is given as the default.

### Model
Here are some example scripts.

LSTM-CharCNN-Large (this is the default: should get ~82 on dev and ~79 on test)
```
th main.lua -gpuid 0 -savefile char-large
```

LSTM-CharCNN-Small (should get ~96 on dev and ~93 on test)
```
th main.lua -gpuid 0 -savefile char-small -rnn_size 300 -highway_layers 1 -kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}'
```

LSTM-Word-Large (should get ~89 on dev and ~85 on test)
```
th main.lua -gpuid 0 -savefile word-large -word_vec_size 650 -highway_layers 0 -use_chars 0 -use_words 1
```

LSTM-Word-Small (should get ~101 on dev and ~98 on test)
```
th main.lua -gpuid 0 -savefile word-small -word_vec_size 200 -highway_layers 0 -use_chars 0 -use_words 1 -rnn_size 200
```
### Licence
MIT



