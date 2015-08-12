# word-char-rnn

large-char (this is the default: should get ~82 on dev and ~79 on test)
th main.lua -gpuid 0 -savefile char-large

small-char (should get ~96 on dev and ~93 on test)
th main.lua -gpuid 0 -savefile char-small -rnn_size 300 -highway_layers 1 -kernels '{1,2,3,4,5,6}' -feature_maps '{25,50,75,100,125,150}'

large-word (should get ~89 on dev and ~85 on test)
th main.lua -gpuid 0 -savefile word-large -word_vec_size 650 -highway_layers 0 -use_chars 0 -use_words 1

small-word (should get ~101 on dev and ~98 on test)
th main.lua -gpuid 0 -savefile word-small -word_vec_size 200 -highway_layers 0 use_chars 0 -use_words 1 -rnn_size 200



