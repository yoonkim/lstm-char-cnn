--[[
model introspection
--]]

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.Squeeze'
require 'util.misc'

BatchLoader = require 'util.BatchLoaderUnk'
model_utils = require 'util.model_utils'
TDNN = require 'model.TDNN'
LSTMTDNN = require 'model.LSTMTDNN'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Perform model introspection')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-model','model.t7', 'model file')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt2 = cmd:parse(arg)

-- load model
checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
torch.manualSeed(opt.seed)
protos = checkpoint.protos
idx2word, word2idx, idx2char, char2idx = table.unpack(opt.vocab)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

function word2char2idx(word)
    local char_idx = torch.zeros(opt.max_word_l)
    char_idx:fill(1) -- fill with padding first
    local l = opt.padding + 1 -- skip beginning padding
    for c in word:gmatch'.' do
        -- while character is valid and we are under max word length
        if char2idx[c] ~= nil and l <= char_idx:size(1) then
	    char_idx[l] = char2idx[c]
	    l = l + 1
	end
    end
end

-- get layers which will be referenced layer (during SGD or introspection)
function get_layer(layer)
    local tn = torch.typename(layer)
    if layer.name ~= nil then
        if layer.name == 'word_vecs' then
	    word_vecs = layer
	elseif layer.name == 'char_vecs' then
	    char_vecs = layer
	elseif layer.name == 'cnn' then
	    cnn = layer
	end
    end
end 
protos.rnn:apply(get_layer)

-- get conv filter layers
conv_filters = {}
cnn:apply(function (x) if x.name ~= nil then if x.name:sub(1,4)=='conv' then 
			  table.insert(conv_filters, x) end end end)

-- for each word get the feature map values as well
-- as the chargrams that activate the feature map (i.e. max)
function get_max_chargrams()
    local result = {}
    local char_idx_all = torch.zeros(#idx2word, opt.max_word_l)
    for i = 1, #idx2word do
        char_idx_all[i] = word2char2idx(opt.START .. word2idx[i] .. opt.END)
    end
    local char_vecs_all = char_vecs:forward(char_idx_all) -- vocab_size x max_word_l x char_vec_size
    for i = 1, #conv_filters do
        local conv_filter = conv_filters[i]
	local width = conv_filter.kW
	result[width] = {}
	local conv_output = conv_filter:forward(char_vecs_all,2)
	local max_val, max_arg = torch.max(conv_output) -- get max values and argmaxes
	max_val = max_val:squeeze()
	max_arg =  max_arg:squeeze()
	result[width][1] = max_val
	result[width][2] = {} -- this is where we'll store the chargrams (as strings)
	for j = 1, #idx2word do
    	    local chargrams = {}
	    for k = 1, max_arg:size(2) do
	        local c = {}
	        local start_char = max_arg[j][k] 
		local end_char = max_arg[j][k] + width - 1
		for l = start_char, end_char do
		    table.insert(c, idx2char[char_idx_all[j][l]])
		end
		chargrams[#chargrams + 1] = table.concat(c)
	    end
	    result[width][2][j] = chargrams
	end
    end
    return result
end

result = get_max_chargrams()
max_chargrams = {}
for u,v in pairs(result) do
    local max_val, max_arg = torch.max(v[1],1)
    max_val = max_val:squeeze()
    max_arg = max_arg:squeeze()
    for i = 1, max_arg:size(1) do -- for each feature map
        local chargram = v[2][max_arg[i]][i]
	max_chargrams[chargram] = max_val[i]
    end
end
-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end


-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

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
function eval_split(split_idx, max_batches, full_eval)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local total_unk_perp = 0
    local unk_perp = 0
    local unk_count = 0
    local rnn_state = {[0] = init_state}    
    if full_eval==nil then -- batch eval        
	for i = 1,n do -- iterate over batches in the split
	    -- fetch a batch
	    local x, y, x_char = loader:next_batch(split_idx)
	    if opt.gpuid >= 0 then -- ship the input arrays to GPU
		-- have to convert to float because integers can't be cuda()'d
		x = x:float():cuda()
		y = y:float():cuda()
		x_char = x_char:float():cuda()
	    end
	    -- forward pass
	    for t=1,opt.seq_length do
		clones.rnn[t]:evaluate() -- for dropout proper functioning
		local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
		rnn_state[t] = {}
		for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
		prediction = lst[#lst] 
		loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
	    end
	    -- carry over lstm state
	    rnn_state[0] = rnn_state[#rnn_state]
	    -- print(i .. '/' .. n .. '...')
	end
	loss = loss / opt.seq_length / n
    else -- full eval on test set
        local x, y, x_char = loader:next_batch(split_idx)
	protos.rnn:evaluate() -- just need one clone
	for t = 1, x:size(2) do
	    local lst = protos.rnn:forward(get_input(x, x_char, t, rnn_state[0]))
	    rnn_state[0] = {}
	    for i=1,#init_state do table.insert(rnn_state[0], lst[i]) end
	    prediction = lst[#lst] 
	    local tok_perp = protos.criterion:forward(prediction, y[{{},t}])
	    loss = loss + tok_perp
	    if x[1][t] == loader.word2idx['|'] then -- count perplexity for <unk> contexts
	        unk_perp = unk_perp + tok_perp
		unk_count = unk_count + 1
	    end
	    -- print(t .. '/' .. unk_perp .. '/' .. unk_count .. '/' .. loss)
	end
	total_unk_perp = torch.exp(unk_perp / unk_count)
	loss = loss / x:size(2)
    end    
    local perp = torch.exp(loss)    
    return perp, total_unk_perp
end

