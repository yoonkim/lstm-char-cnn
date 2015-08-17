## Neural Language Modeling with Characters
A neural language model (NLM) built on character inputs only. 
The model employs a convolutional neural network (CNN) over character
embeddings to use as inputs into an long short-term memory (LSTM)
recurrent neural network language model (RNNLM). Also optionally
passes the output from the CNN through a [Highway Network](http://arxiv.org/abs/1507.06228), 
which improves performance.

Note: Code is messy/experimental. Cleaner (and faster) code coming. Paper 
will be posted on arXiv very soon.

Much of the base code is from Andrej Karpathy's excellent character RNN implementation,
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

`cudnn` also will result in a good (10x) speed-up.

### Data
Data should be put into the `data/` directory, split into `train.txt`,
`valid.txt`, and `test.txt`

Each line of the .txt file should be a sentence. The English Penn 
Treebank data (Tomas Mikolov's pre-processed version with vocab size equal to 10K,
widely used by the language modeling community) is given as the default.

### Model
Here are some example scripts. Add `-gpuid 0` to use a GPU (which is
required to get any reasonable speed with the CNN)

Large character-level model (`LSTM-CharCNN-Large` in the paper).
This is the default: should get ~82 on valid and ~79 on test.
```
th main.lua -savefile char-large
```

Small character-level model (`LSTM-CharCNN-Small` in the paper).
This should get ~96 on valid and ~93 on test.
```
th main.lua -savefile char-small -rnn_size 300 -highway_layers 1 
-kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}'
```

Large word-level model (`LSTM-Word-Large` in the paper).
This should get ~89 on valid and ~85 on test.
```
th main.lua -savefile word-large -word_vec_size 650 -highway_layers 0 
-use_chars 0 -use_words 1
```

Small word-level model (`LSTM-Word-Small` in the paper).
This should get ~101 on valid and ~98 on test.
```
th main.lua -savefile word-small -word_vec_size 200 -highway_layers 0 
-use_chars 0 -use_words 1 -rnn_size 200
```

Note that if `-use_chars` and `-use_words` are both set to 1, the model
will concatenate the output from the CNN with the word embedding. We've
found this model to underperform a purely character-level model, though.

### Licence
MIT



