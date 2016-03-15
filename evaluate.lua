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
require 'util.misc'
require 'util.HLogSoftMax'

HSMClass = require 'util.HSMClass'
BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain train.txt/valid.txt/test.txt with input data')
cmd:option('-savefile', 'model_results.t7', 'save results to')
cmd:option('-model', 'en-large-word-model.t7', 'model checkpoint file')
-- GPU/CPU these params must be passed in because it affects the constructors
cmd:option('-gpuid', -1,'which gpu to use. -1 = use CPU')
cmd:option('-cudnn', 0,'use cudnn (1 = yes, 0 = no)')

cmd:text()

-- parse input params
opt2 = cmd:parse(arg)
if opt2.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt2.gpuid + 1)
end

if opt2.cudnn == 1 then
    assert(opt2.gpuid >= 0, 'GPU must be used if using cudnn')
    print('using cudnn')
    require 'cudnn'
end

HighwayMLP = require 'model.HighwayMLP'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
protos = checkpoint.protos
print('opt: ')
print(opt)
print('val_losses: ')
print(checkpoint.val_losses)
idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)

-- recreate the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding, opt.max_word_l)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char
	    .. ', Max word length (incl. padding): ', loader.max_word_l)

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(2, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

params, grad_params = model_utils.combine_all_parameters(protos.rnn)
if opt.hsm > 0 then
    hsm_params, hsm_grad_params = model_utils.combine_all_parameters(protos.criterion)
    print('number of parameters in the model: ' .. params:nElement() + hsm_params:nElement())
else
    print('number of parameters in the model: ' .. params:nElement())
end

-- for easy switch between using words/chars (or both)
function get_input(x, x_char, t, prev_states)
    local u = {}
    if opt.use_chars == 1 then 
        table.insert(u, x_char[{{1,2},t}])
    end
    if opt.use_words == 1 then 
        table.insert(u, x[{{1,2},t}]) 
    end
    for i = 1, #prev_states do table.insert(u, prev_states[i]) end
    return u
end

-- evaluate the loss over an entire split
function eval_split_full(split_idx)
    print('evaluating loss over split index ' .. split_idx)
    if opt.hsm > 0 then
        protos.criterion:change_bias()
    end
    local n = loader.split_sizes[split_idx]
    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local token_count = torch.zeros(#idx2word)
    local token_loss = torch.zeros(#idx2word)
    local rnn_state = {[0] = init_state}    
    local x, y, x_char = loader:next_batch(split_idx)
    if opt.gpuid >= 0 then
        x = x:float():cuda()
	y = y:float():cuda()
	x_char = x_char:float():cuda()
    end
    protos.rnn:evaluate() 
    for t = 1, x:size(2) do
	local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
	rnn_state[0] = {}
	for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	prediction = lst[#lst] 
	local singleton_loss = protos.criterion:forward(prediction, y[{{1,2},t}])
	loss = loss + singleton_loss
	local token_idx = x[1][t]
	token_count[token_idx] = token_count[token_idx] + 1
	token_loss[token_idx] = token_loss[token_idx] + singleton_loss
    end
    loss = loss / x:size(2)
    local total_perp = torch.exp(loss)    
    return total_perp, token_loss:float(), token_count:float()
end

total_perp, token_loss, token_count = eval_split_full(3)
print(total_perp)
test_results = {}
test_results.perp = total_perp
test_results.token_loss = token_loss
test_results.token_count = token_count
test_results.vocab = {idx2word, word2idx, idx2char, char2idx}
test_results.opt = opt
test_results.val_losses = checkpoint.val_losses
torch.save(opt2.savefile, test_results)
collectgarbage()
