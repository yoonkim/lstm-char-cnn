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
cmd:text('Train a word+character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-model','model.t7', 'model file')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt2 = cmd:parse(arg)
torch.manualSeed(opt.seed)
checkpoint = torch.load(opt2.model)
opt = checkpoint.opt
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
    char_idx:fill(1]) -- fill with padding first
    local l = opt.padding + 1 -- skip beginning padding
    for c in word:gmatch'.' do
        -- while character is valid and we are under max word length
        if char2idx[c] ~= nil and l <= char_idx:size(1) then
	    char_idx[l] = char2idx[c]
	    l = l + 1
	end
    end
end

-- recreate the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.padding)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char)

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

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

print('number of parameters in the model: ' .. params:nElement())

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

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, x_char = loader:next_batch(1) --from train
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
	x_char = x_char:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward(get_input(x, x_char, t, rnn_state[t-1]))
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
	table.insert(rnn_state[t-1], drnn_state[t])
        local dlst = clones.rnn[t]:backward(get_input(x, x_char, t, rnn_state[t-1]), drnn_state[t])
        drnn_state[t-1] = {}
	local tmp = opt.use_words + opt.use_chars -- not the safest way but quick
        for k,v in pairs(dlst) do
            if k > tmp then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-tmp] = v
            end
        end	
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?

    -- renormalize gradients
    local grad_norm = grad_params:norm()
    if grad_norm > opt.max_grad_norm then
        local shrink_factor = opt.max_grad_norm / grad_norm
	grad_params:mul(shrink_factor)
    end    
    params:add(grad_params:mul(-lr)) -- update params
    return loss
end

-- start optimization here
train_losses = {}
val_losses = {}
lr = opt.learning_rate -- starting learning rate which will be decayed
local iterations = opt.max_epochs * loader.split_sizes[1]
for i = 1, iterations do
    local epoch = i / loader.split_sizes[1]

    local timer = torch.Timer()
    local time = timer:time().real

    train_loss = feval(params) -- fwd/backprop and update params
    if char_vecs ~= nil then char_vecs.weight[1]:zero() end -- zero-padding vector is always zero
    train_losses[i] = train_loss

    -- decay learning rate after epoch
    if i % loader.split_sizes[1] == 0 and epoch >= opt.learning_rate_decay_after then        
        lr = lr * opt.learning_rate_decay        
    end    

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations or i % loader.split_sizes[1] == 0 then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.2f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = {loader.idx2word, loader.word2idx, loader.idx2char, loader.char2idx}
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.2f), train_loss = %6.4f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end   
    if i % 10 == 0 then collectgarbage() end
end


