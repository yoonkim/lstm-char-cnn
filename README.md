## Character-Aware Neural Language Models
Code for the paper [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615).

A neural language model (NLM) built on character inputs only. Predictions
are still made at the word-level. The model employs a convolutional neural network (CNN)
over characters to use as inputs into an long short-term memory (LSTM)
recurrent neural network language model (RNN-LM). Also optionally
passes the output from the CNN through a [Highway Network](http://arxiv.org/abs/1507.06228), 
which improves performance.

Much of the base code is from 
[Andrej Karpathy's excellent character RNN implementation](https://github.com/karpathy/char-rnn).

### Requirements
Code is written in Lua and requires Torch. It also requires
the `nngraph` and the `luautf8` packages, which can be installed via:
```
luarocks install nngraph
luarocks install luautf8
```
GPU usage will additionally require `cutorch` and `cunn` packages:
```
luarocks install cutorch
luarocks install cunn
```

`cudnn` will result in a good (8x-10x) speed-up for convolutions, so it is
highly recommended. This will make the training time of a character-level model 
be somewhat competitive against a word-level model (1500 tokens/sec vs 3000 tokens/sec for 
the large character/word-level models described below).

```
git clone https://github.com/soumith/cudnn.torch.git
cd cudnn.torch
luarocks make cudnn-scm-1.rockspec
```
### Data
Data should be put into the `data/` directory, split into `train.txt`,
`valid.txt`, and `test.txt`

Each line of the .txt file should be a sentence. The English Penn 
Treebank (PTB) data (Tomas Mikolov's pre-processed version with vocab size equal to 10K,
widely used by the language modeling community) is given as the default.

The paper also runs the models on non-English data (Czech, French, German, Russian, and Spanish), from the ICML 2014
paper [Compositional Morphology for Word Representations and Language Modelling](http://arxiv.org/abs/1405.4273)
by Jan Botha and Phil Blunsom. This can be downloaded from [Jan's website](https://bothameister.github.io).

For ease of use, we provide a script to download the non-English data (`get_data.sh`). 
The script also saves the downloaded data into the relevant folders.

#### Note on PTB
The PTB data above does not have end-of-sentence tokens for each sentence, and hence these must be
manually appended. This can be done by adding `-EOS '+'` to the script (obviously you 
can use other characters than `+` to represent an end-of-sentence token---we recommend a single
unused character).

The non-English data already have end-of-sentence tokens for each line so you do not need to 
add the `-EOS` command (equivalent to adding `-EOS ''`, which is the default).

#### Unicode in Lua
Lua is unicode-agnostic (each string is just a sequence of bytes) so we use
the `luautf8` package to deal with languages where a character can be more than one byte
(e.g. Russian). Many thanks to [vseledkin](https://github.com/vseledkin) for alerting us
to the fact that previous version of the code did not take this account!

### Model
Here are some example scripts. Add `-gpuid 0` to each line to use a GPU (which is
required to get any reasonable speed with the CNN), and `-cudnn 1` to use the
cudnn package. Scripts to reproduce the results of the paper can be found under `run_models.sh`

#### Character-level models
Large character-level model (LSTM-CharCNN-Large in the paper).
This is the default: should get ~82 on valid and ~79 on test.
```
th main.lua -savefile char-large -EOS '+'
```
Small character-level model (LSTM-CharCNN-Small in the paper).
This should get ~96 on valid and ~93 on test.
```
th main.lua -savefile char-small -rnn_size 300 -highway_layers 1 
-kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}' -EOS '+'
```

#### Word-level models
Large word-level model (LSTM-Word-Large in the paper).
This should get ~89 on valid and ~85 on test.
```
th main.lua -savefile word-large -word_vec_size 650 -highway_layers 0 
-use_chars 0 -use_words 1 -EOS '+'
```
Small word-level model (LSTM-Word-Small in the paper).
This should get ~101 on valid and ~98 on test.
```
th main.lua -savefile word-small -word_vec_size 200 -highway_layers 0 
-use_chars 0 -use_words 1 -rnn_size 200 -EOS '+'
```

#### Combining both
Note that if `-use_chars` and `-use_words` are both set to 1, the model
will concatenate the output from the CNN with the word embedding. We've
found this model to underperform a purely character-level model, though.

### Evaluation
By default `main.lua` will evaluate the model on test data after training,
but this will use the last epoch's model, and also will be slow due to
the way the data is set up.

Evaluation on test can be performed via the following script:
```
th evaluate.lua -model model_file.t7 -data_dir data/ptb -savefile model_results.t7
```
Where `model_file.t7` is the path to the best performing (on validation) model.
This will also save some basic statistics (e.g. perplexity by token) in
`model_results.t7`.

### Hierarchical Softmax
Training on a larger vocabulary (e.g. 100K+) will require hierarchical softmax (HSM)
to train at a reasonable speed. You can use the `-hsm` option to do this.
For example `-hsm 500` will randomly split the vocabulary into 500 clusters of
(approximately) equal size. `-hsm 0` is the default and will not use HSM.
`-hsm -1` will automatically choose the number of clusters for you, by choosing the integer
closest to sqrt(|V|).

### Batch Size
If training on bigger datasets you should probably use a 
larger batch size (e.g. `-batch_size 100`).

### Licence
MIT



