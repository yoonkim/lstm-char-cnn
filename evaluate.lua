--[[
Evaluates a trained model

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'
TDNN = require 'model.AdaTDNN'
LSTMTDNN = require 'model.LSTMTDNN'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tiger','data directory. Should contain train.txt/valid.txt/test.txt with input data')
-- model params
cmd:option('-model', 'cv/lm_char-attend_epoch8.00_181.62.t7', 'model checkpoint file')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt2 = cmd:parse(arg)
checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
protos = checkpoint.protos
idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

-- recreate the data loader class, with batchsize = 1
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding, opt.max_word_l)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
	    .. ', Max word length (incl. padding): ', loader.max_word_l)

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(1, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- training criterion (negative log likelihood)
protos.criterion = nn.ClassNLLCriterion()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
params:uniform(-0.05, 0.05) -- small numbers uniform

print('number of parameters in the model: ' .. params:nElement())

-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
    local u = {}
    if opt.use_chars == 1 then table.insert(u, x_char[{{},t}]) end
    if opt.use_words == 1 then table.insert(u, x[{{},t}]) end
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end

-- evaluate the loss over an entire split
function eval_split_full(split_idx)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local token_count = torch.zeros(#idx2word)
    local token_loss = torch.zeros(#idx2word)
    local rnn_state = {[0] = init_state}    
    local x, y, x_char = loader:next_batch(split_idx)
    protos.rnn:evaluate() 
    for t = 1, x:size(2) do
	local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
	rnn_state[0] = {}
	for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	prediction = lst[#lst] 
	local singleton_loss = protos.criterion:forward(prediction, y[{{},t}])
	loss = loss + singleton_loss
	local token_idx = x[1][t]
	token_count[token_idx] = token_count[token_idx] + 1
	token_loss[token_idx] = token_loss[token_idx] + singleton_loss
    end
    loss = loss / x:size(2)
    local total_perp = torch.exp(loss)    
    return total_perp, token_loss, token_count
end

collectgarbage()