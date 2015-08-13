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

HighwayMLP = require 'model.HighwayMLP'
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
cmd:option('-savefile', 'chargrams.tsv', 'save max chargrams to')
cmd:text()

-- parse input params
opt2 = cmd:parse(arg)

if opt2.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt2.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt2.gpuid + 1)
end


-- load model
checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
torch.manualSeed(opt.seed)
protos = checkpoint.protos
idx2word, word2idx, idx2char, char2idx = table.unpack(checkpoint.vocab)


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
    return char_idx
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
	elseif layer.name == 'highway' then
	    highway = layer
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
        char_idx_all[i] = word2char2idx(opt.tokens.START .. idx2word[i] .. opt.tokens.END)
    end
    local char_vecs_all = char_vecs:forward(char_idx_all) -- vocab_size x max_word_l x char_vec_size
    for i = 1, #conv_filters do
    	local max_val, max_arg
        local conv_filter = conv_filters[i]
	local width = conv_filter.kW
	result[width] = {}
	local conv_output = conv_filter:forward(char_vecs_all)
	max_val, max_arg = torch.max(conv_output,2) -- get max values and argmaxes
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
	local word = idx2word[max_arg[i]]
	max_chargrams[#max_chargrams + 1] = {u, chargram, word, max_val[i]}
    end
end

f = io.open(opt2.savefile, 'w')
for u,v in ipairs(max_chargrams) do
    f:write(v[1]..'\t'..v[2]..'\t'..v[3]..'\t'..v[4]..'\n')
end
f:close()