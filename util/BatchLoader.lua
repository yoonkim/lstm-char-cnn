
-- Modified from https://github.com/karpathy/char-rnn
-- This version is for cases where one has already segmented train/val/test splits

local BatchLoader = {}
local stringx = require('pl.stringx')
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_dir, batch_size, seq_length, padding, max_word_l)
    local self = {}
    setmetatable(self, BatchLoader)
    self.padding = padding or 0
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local input_files = {train_file, valid_file, test_file}
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')

    -- construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file)) then
        print('one-time setup: preprocessing input train/valid/test files in dir: ' .. data_dir)
        BatchLoader.text_to_tensor(input_files, vocab_file, tensor_file)
    end

    print('loading data files...')
    local all_data = torch.load(tensor_file) -- train, valid, test tensors
    local vocab_mapping = torch.load(vocab_file)
    self.idx2word = vocab_mapping[1]; self.word2idx = vocab_mapping[2]
    self.idx2char = vocab_mapping[3]; self.char2idx = vocab_mapping[4]
    self.idx2char[#self.idx2char + 1] = '{'; self.char2idx['{'] = #self.idx2char -- start-of-word symbol
    self.idx2char[#self.idx2char + 1] = '}'; self.char2idx['}'] = #self.idx2char -- end-of-word symbol
    self.idx2char[#self.idx2char + 1] = ' '; self.char2idx[' '] = #self.idx2char -- zero-padding
    self.vocab_size = #self.idx2word
    print(string.format('Word vocab size: %d, Char vocab size: %d', #self.idx2word, #self.idx2char))

    -- create word-char mappings
    if max_word_l == nil then -- if max word length is not specified
	self.max_word_l = 0
	for i = 1, #self.idx2word do
	    self.max_word_l = math.max(self.max_word_l, self.idx2word[i]:len()) -- get max word length 
	end
    else
        self.max_word_l = max_word_l
    end
    self.word2char2idx = torch.zeros(#self.idx2word, self.max_word_l + 2*self.padding):long()
    for i = 1, #self.idx2word do
        self.word2char2idx[i] = self:get_word2char2idx("{"..self.idx2word[i].."}")
    end
    -- cut off the end for train/valid sets so that it divides evenly
    -- test set is not cut off
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.split_sizes = {}
    self.all_batches = {}
    local lookup = nn.LookupTableInt(self.word2char2idx)
    print('reshaping tensors...')  
    local x_batches, y_batches, nbatches
    for split, data in ipairs(all_data) do
    	local len = data:size(1)
	if len % (batch_size * seq_length) ~= 0 and split < 3 then
	    data = data:sub(1, batch_size * seq_length * math.floor(len / (batch_size * seq_length)))
	end
	local ydata = data:clone()
	ydata:sub(1,-2):copy(data:sub(2,-1))
	ydata[-1] = data[1]
	if split < 3 then
	    x_batches = data:view(batch_size, -1):split(seq_length, 2)
	    y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
	    nbatches = #x_batches	   
	    self.split_sizes[split] = nbatches
	    assert(#x_batches == #y_batches)
	else --for test we repeat dimensions to batch size for easier evaluation
	    x_batches = {data:resize(1, data:size(1)):expand(batch_size, data:size(2))}
	    y_batches = {ydata:resize(1, ydata:size(1)):expand(batch_size, ydata:size(2))}
	    self.split_sizes[split] = 1	
	end
	local x_char_batches = {}
	for i = 1, #x_batches do
	    x_char_batches[#x_char_batches + 1] = lookup:forward(x_batches[i]):clone()
	end
  	self.all_batches[split] = {x_batches, y_batches, x_char_batches}
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoader:get_word2char2idx(word)
    local char_idx = torch.zeros(self.max_word_l+ 2*self.padding)
    char_idx:fill(self.char2idx[' ']) -- fill with padding first
    local l = self.padding + 1
    for c in word:gmatch'.' do
        if self.char2idx[c] ~= nil and l <= char_idx:size(1) then -- cutoff if word is too long
	    char_idx[l] = self.char2idx[c]
	    l = l + 1
	end
    end
    return char_idx:long()
end

function BatchLoader:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoader:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    return self.all_batches[split_idx][3][idx], self.all_batches[split_idx][2][idx], self.all_batches[split_idx][3][idx]
end

function BatchLoader.text_to_tensor(input_files, out_vocabfile, out_tensorfile)
    print('Processing text into tensors...')
    local f, rawdata, output
    local output_tensors = {} -- output tensors for train/val/test
    local vocab_count = {} -- vocab count 
    local idx2word = {}; local word2idx = {}
    local idx2char = {}; local char2idx = {}
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
        output = {}
        f = torch.DiskFile(input_files[split])
	rawdata = f:readString('*a') -- read all data at once
	f:close()
	rawdata = stringx.replace(rawdata, '\n', '+') -- use '+' instead of '<eos>' for end-of-sentence
	rawdata = stringx.replace(rawdata, '{', ' ') -- '{' is reserved for start-of-word symbol
	rawdata = stringx.replace(rawdata, '}', ' ') -- '}' is reserved for end-of-word symbol
	rawdata = stringx.replace(rawdata, '<unk>', '|') -- '<unk>' gets replaced with a single character
	for word in rawdata:gmatch'([^%s]+)' do
	    if word2idx[word]==nil then
	        idx2word[#idx2word + 1] = word -- create word-idx/idx-word mappings
		word2idx[word] = #idx2word
		for char in word:gmatch'.' do
		    if char2idx[char]==nil then
		        idx2char[#idx2char + 1] = char -- create char-idx/idx-char mappings
			char2idx[char] = #idx2char
		    end
		end
	    end
	    output[#output + 1] = word2idx[word]
	end	
	output_tensors[split] = torch.LongTensor(output)
    end

    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx, idx2char, char2idx})
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensors)
end

return BatchLoader

