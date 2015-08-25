#!/bin/bash

#run get_data.sh to get all the relevant data
#add -gpuid 0 to use GPU 
#add -cudnn 1 to use cudnn 

#To reproduce Table 3:
#LSTM-Word-Small
th main.lua -data_dir data/ptb -savefile ptb-word-small -EOS '+' -rnn_size 200 -use_chars 0 
-use_words 1 -word_vec_size 200 -highway_layers 0 
#LSTM-CharCNN-Small
th main.lua -data_dir data/ptb -savefile ptb-char-small -EOS '+' -rnn_size 300 -use_chars 0 
-use_words 1 -char_vec_size 15 -highway_layers 1 -kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}'
#LSTM-Word-Large
th main.lua -data_dir data/ptb -savefile ptb-word-large -EOS '+' -rnn_size 650 -use_chars 0 
-use_words 1 -word_vec_size 650 -highway_layers 0
#LSTM-CharCNN-Large
th main.lua -data_dir data/ptb -savefile ptb-char-large -EOS '+' -rnn_size 650 -use_chars 1 -use_words 0 
-char_vec_size 15 -highway_layers 2 -kernels '{1,2,3,4,5,6,7}' -feature_maps '{50,100,150,200,200,200,200}'

#To reproduce Table 4/Table 5, run the same scripts as above but change data_dir/savefile, 
#and remove -EOS. So for German (DE), use the following scripts
#LSTM-Word-Small
th main.lua -data_dir data/de -savefile de-word-small -rnn_size 200 -use_chars 0 
-use_words 1 -word_vec_size 200 -highway_layers 0 
#LSTM-CharCNN-Small
th main.lua -data_dir data/ptb -savefile de-char-small -rnn_size 300 -use_chars 0 
-use_words 1 -char_vec_size 15 -highway_layers 1 -kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}'
#LSTM-Word-Large
th main.lua -data_dir data/ptb -savefile de-word-large -rnn_size 650 -use_chars 0 
-use_words 1 -word_vec_size 650 -highway_layers 0
#LSTM-CharCNN-Large
th main.lua -data_dir data/ptb -savefile de-char-large -rnn_size 650 -use_chars 1 -use_words 0 
-char_vec_size 15 -highway_layers 2 -kernels '{1,2,3,4,5,6,7}' -feature_maps '{50,100,150,200,200,200,200}'
