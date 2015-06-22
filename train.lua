--[[
Trains a word+character-level multi-layer rnn language model

Much of the code is borrowed from the following implementations
https://github.com/karpathy/char-rnn
https://github.com/wojzaremba/lstm
]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'util.OneHot'
require 'util.Squeeze'
require 'util.LookupTableInt'
require 'util.misc'
require 'util.OuterProduct'

local BatchLoader = require 'util.BatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local LSTMCNN = require 'model.LSTMCNN2'

local stringx = require('pl.stringx')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a word+character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 200, 'size of LSTM internal state')
cmd:option('-word_vec_size', 30, 'dimensionality of word embeddings')
cmd:option('-char_vec_size', 30, 'dimensionality of character embeddings')
cmd:option('-feature_maps', '{5,5,5}', 'number of feature maps in the CNN')
cmd:option('-kernels', '{2,3,4}', 'conv net kernel widths')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'for now only lstm is supported. keep fixed')
-- optimization
cmd:option('-learning_rate',1,'starting learning rate')
cmd:option('-learning_rate_decay',0.5,'learning rate decay')
cmd:option('-learning_rate_decay_after',4,'in number of epochs, when to start decaying the learning rate')
cmd:option('-dropout',0,'dropout to use just before classifier. 0 = no dropout')
cmd:option('-seq_length',20,'number of timesteps to unroll for')
cmd:option('-batch_size',20,'number of sequences to train on in parallel')
cmd:option('-max_epochs',14,'number of full passes through the training data')
cmd:option('-max_grad_norm',5,'normalize gradients at')
-- bookkeeping
cmd:option('-seed',3435,'torch manual random number generator seed')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',30000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1)
end

loadstring("kernels = " .. opt.kernels)() -- get kernel sizes
local padding = torch.Tensor(kernels):max()-1 -- padding is max kernel size minus one
loadstring("feature_maps = " .. opt.feature_maps)() -- get feature map sizes

-- create the data loader class
loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, padding)
print('Word vocab size: ' .. #loader.idx2word .. ', Char vocab size: ' .. #loader.idx2char)

-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end


-- define the model: prototypes for one timestep, then clone them in time
protos = {}
print('creating an LSTM-CNN with ' .. opt.num_layers .. ' layers')
--protos.rnn = LSTM.lstm(#loader.idx2word, opt.rnn_size, opt.num_layers, opt.dropout)
protos.rnn = LSTMCNN.lstmcnn(#loader.idx2word, opt.rnn_size, opt.num_layers, opt.dropout, opt.word_vec_size,
	         opt.char_vec_size, #loader.idx2char, feature_maps, kernels, loader.word2char2idx)
-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
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

-- get certain layers which will may be manipulated separately during SGD
function get_layer(layer)
    local tn = torch.typename(layer)
    if tn == 'nn.LookupTable' then
        if layer.weight:size(1) == #loader.idx2word then
	    word_vecs = layer
	elseif layer.weight:size(1) == #loader.idx2char then
	    char_vecs = layer
	end
    end
end 
protos.rnn:apply(get_layer)

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_idx, max_batches)
    print('evaluating loss over split index ' .. split_idx)
    local n = loader.split_sizes[split_idx]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_idx) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    if split_idx < 3 then --evaluation is different btw test vs train/val
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
		local lst = clones.rnn[t]:forward{x[{{}, t}], x_char[{{},t}], unpack(rnn_state[t-1])}
		rnn_state[t] = {}
		for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
		prediction = lst[#lst] 
		loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
	    end
	    -- carry over lstm state
	    rnn_state[0] = rnn_state[#rnn_state]
	    print(i .. '/' .. n .. '...')
	end
    end
    loss = loss / opt.seq_length / n
    local perp = torch.exp(loss)
    return perp
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
        local lst = clones.rnn[t]:forward{x[{{}, t}], x_char[{{},t}], unpack(rnn_state[t-1])}
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
        local dlst = clones.rnn[t]:backward({x[{{}, t}], x_char[{{},t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 2 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-2] = v
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
    char_vecs.weight[#loader.idx2char]:zero() -- zero-padding vector is always zero
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

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
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
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end
end


