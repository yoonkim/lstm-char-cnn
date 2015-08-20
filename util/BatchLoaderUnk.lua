
-- Modified from https://github.com/karpathy/char-rnn
-- This version is for cases where one has already segmented train/val/test splits

local BatchLoaderUnk = {}
local stringx = require('pl.stringx')
BatchLoaderUnk.__index = BatchLoaderUnk



function BatchLoaderUnk.create(data_dir, batch_size, seq_length, padding, max_word_l, max_factor_l, use_morpho)
    local self = {}
    setmetatable(self, BatchLoaderUnk)

    self.padding = padding or 0
    local train_file = path.join(data_dir, 'train.txt')
    local valid_file = path.join(data_dir, 'valid.txt')
    local test_file = path.join(data_dir, 'test.txt')
    local in_morpho_file = path.join(data_dir, 'morpho.txt')
    local input_files = {train_file, valid_file, test_file}
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')
    local char_file = path.join(data_dir, 'data_char.t7')
    local out_morpho_file = path.join(data_dir, 'morpho.t7')


    -- construct a tensor with all the data
    if not (path.exists(vocab_file) or path.exists(tensor_file) or path.exists(char_file)) then
        print('one-time setup: preprocessing input train/valid/test files in dir: ' .. data_dir)
        BatchLoaderUnk.text_to_tensor(input_files, in_morpho_file, use_morpho, vocab_file, tensor_file, char_file, 
                                      out_morpho_file,
                                      max_word_l, max_factor_l)
    end

    print('loading data files...')
    local all_data = torch.load(tensor_file) -- train, valid, test tensors
    local all_data_char = torch.load(char_file) -- train, valid, test character indices
    local all_data_morpho = torch.load(out_morpho_file) -- train, valid, test character indices
    local vocab_mapping = torch.load(vocab_file)
    self.idx2word, self.word2idx, self.idx2char, self.char2idx, self.idx2morpho, self.morpho2idx = table.unpack(vocab_mapping)
    self.vocab_size = #self.idx2word
    print(string.format('Word vocab size: %d, Char vocab size: %d', #self.idx2word, #self.idx2char))
    -- create word-char mappings
    self.max_word_l = all_data_char[1]:size(2)
    -- cut off the end for train/valid sets so that it divides evenly
    -- test set is not cut off
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.split_sizes = {}
    self.all_batches = {}
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
       local data_char = torch.zeros(data:size(1), self.max_word_l):long()
       for i = 1, data:size(1) do
          data_char[i] = self:expand(all_data_char[split][i]:totable())
       end

       local data_morpho
       if use_morpho then
          data_morpho = torch.ones(data:size(1), max_factor_l):long()
          for i = 1, data:size(1) do
             data_morpho[i] = all_data_morpho[split][i]
          end
       end

       if split < 3 then
          x_batches = data:view(batch_size, -1):split(seq_length, 2)
          y_batches = ydata:view(batch_size, -1):split(seq_length, 2)
          x_char_batches = data_char:view(batch_size, -1, self.max_word_l):split(seq_length,2)
          if use_morpho then
             x_morpho_batches = data_morpho:view(batch_size, -1, max_factor_l):split(seq_length,2)
          end
          nbatches = #x_batches	   
          self.split_sizes[split] = nbatches
          assert(#x_batches == #y_batches)
          assert(#x_batches == #x_char_batches)
       else --for test we repeat dimensions to batch size (easier but inefficient evaluation)
          x_batches = {data:resize(1, data:size(1)):expand(batch_size, data:size(2))}
          y_batches = {ydata:resize(1, ydata:size(1)):expand(batch_size, ydata:size(2))}
          data_char = data_char:resize(1, data_char:size(1), data_char:size(2))
          if use_morpho then
             data_morpho = data_morpho:resize(1, data_morpho:size(1), data_morpho:size(2))
             x_morpho_batches = {data_morpho:expand(batch_size, data_morpho:size(2), data_morpho:size(3))}
          end
          x_char_batches = {data_char:expand(batch_size, data_char:size(2), data_char:size(3))}

          self.split_sizes[split] = 1
       end
       if use_morpho then 
          self.all_batches[split] = {x_batches, y_batches, x_morpho_batches}
       else
          self.all_batches[split] = {x_batches, y_batches, x_char_batches}
       end
    end
    self.batch_idx = {0,0,0}
    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.split_sizes[1], self.split_sizes[2], self.split_sizes[3]))
    collectgarbage()
    return self
end

function BatchLoaderUnk:expand(t)    
    for i = 1, self.padding do
        table.insert(t, 1, 1) -- 1 is always char idx for zero pad
    end
    while #t < self.max_word_l do
        table.insert(t, 1)
    end
    return torch.LongTensor(t):sub(1, self.max_word_l)
end

function BatchLoaderUnk:reset_batch_pointer(split_idx, batch_idx)
    batch_idx = batch_idx or 0
    self.batch_idx[split_idx] = batch_idx
end

function BatchLoaderUnk:next_batch(split_idx)
    -- split_idx is integer: 1 = train, 2 = val, 3 = test
    self.batch_idx[split_idx] = self.batch_idx[split_idx] + 1
    if self.batch_idx[split_idx] > self.split_sizes[split_idx] then
        self.batch_idx[split_idx] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local idx = self.batch_idx[split_idx]
    return self.all_batches[split_idx][1][idx], self.all_batches[split_idx][2][idx], self.all_batches[split_idx][3][idx]
end

function BatchLoaderUnk.text_to_tensor(input_files, morpho_file, use_morpho,
                                       out_vocabfile, out_tensorfile, out_charfile, 
                                       out_morphofile,
                                       max_word_l, max_factor_l)
    print('Processing text into tensors...')
    local tokens = opt.tokens -- inherit global constants for tokens
    local f, rawdata
    local output_tensors = {} -- output tensors for train/val/test
    local output_chars = {} -- output character tensors for train/val/test sets
    local output_morphos = {} 
    local vocab_count = {} -- vocab count 
    local max_word_l_tmp = 0 -- max word length of the corpus
    local idx2word = {tokens.UNK} -- unknown word token
    local word2idx = {}; word2idx[tokens.UNK] = 1
    local idx2char = {tokens.ZEROPAD, tokens.START, tokens.END} -- zero-pad, start-of-word, end-of-word tokens
    local char2idx = {}; char2idx[tokens.ZEROPAD] = 1; char2idx[tokens.START] = 2; char2idx[tokens.END] = 3
    local idx2factor = {tokens.ZEROPAD, tokens.START, tokens.END} -- zero-pad, start-of-word, end-of-word tokens
    local factor2idx = {}; factor2idx[tokens.ZEROPAD] = 1; factor2idx[tokens.START] = 2; factor2idx[tokens.END] = 3
    local split_counts = {}
    local morpho_dict = {}

    if use_morpho then 
       f = io.open(morpho_file, 'r')
       for line in f:lines() do
          local n = 0
          for factor in line:gmatch'([^%s]+)' do
             local word = nil
             if n == 0 then
                word = factor
                if word2idx[word] == nil then
                idx2word[#idx2word + 1] = word
                word2idx[word] = #idx2word
                end
                wordidx = word2idx[word]
                morpho_dict[wordidx] = torch.ones(max_factor_l)
             else
                if factor2idx[factor] == nil then
                   idx2factor[#idx2factor + 1] = factor
                   factor2idx[factor] = #idx2factor
                end
                morpho_dict[wordidx][n] = factor2idx[factor]
             end
             n = n + 1
          end
       end
    end

    -- first go through train/valid/test to get max word length
    -- if actual max word length (e.g. 19 for PTB) is smaller than specified
    -- we use that instead. this is inefficient, but only a one-off thing so should be fine
    -- also counts the number of tokens
    for	split = 1,3 do -- split = 1 (train), 2 (val), or 3 (test)
       f = io.open(input_files[split], 'r')       
       local counts = 0
       for line in f:lines() do
          line = stringx.replace(line, '<unk>', tokens.UNK) -- replace unk with a single character
	  line = stringx.replace(line, tokens.START, '') --start-of-word token is reserved
	  line = stringx.replace(line, tokens.END, '') --end-of-word token is reserved
          for word in line:gmatch'([^%s]+)' do
	     max_word_l_tmp = math.max(max_word_l_tmp, word:len())
	     counts = counts + 1
          end
	  if tokens.EOS ~= '' then
	      counts = counts + 1 --PTB uses \n for <eos>, so need to add one more token at the end
	  end
       end
       f:close()
       split_counts[split] = counts
    end
      
    print('After first pass of data, max word length is: ' .. max_word_l_tmp)
    print(string.format('Token count: train %d, val %d, test %d', 
    			split_counts[1], split_counts[2], split_counts[3]))

    -- if actual max word length is less than the limit, use that
    max_word_l = math.min(max_word_l_tmp, max_word_l)
   
    for	split = 1, 3 do -- split = 1 (train), 2 (val), or 3 (test)     
       -- Preallocate the tensors we will need.
       -- Watch out the second one needs a lot of RAM.
       output_tensors[split] = torch.LongTensor(split_counts[split])
       output_chars[split] = torch.ones(split_counts[split], max_word_l):long()
       if use_morpho then 
          output_morphos[split] = torch.ones(split_counts[split], max_factor_l):long()
       end

       f = io.open(input_files[split], 'r')
       local word_num = 0
       for line in f:lines() do
          line = stringx.replace(line, '<unk>', tokens.UNK)
	  line = stringx.replace(line, tokens.START, '') -- start and end of word tokens are reserved
	  line = stringx.replace(line, tokens.END, '')
          for rword in line:gmatch'([^%s]+)' do
             function append(word)
                word_num = word_num + 1
                -- Collect garbage.
                if word_num % 10000 == 0 then
                   collectgarbage()
                end
                local chars = {char2idx[tokens.START]} -- start-of-word symbol
                if string.sub(word,1,1) == tokens.UNK and word:len() > 1 then -- unk token with character info available
                   word = string.sub(word, 3)
                   output_tensors[split][word_num] = word2idx[tokens.UNK]
                   if use_morpho then 
                      output_morphos[split][word_num] = morpho_dict[word2idx[tokens.UNK]]
                   end
                else
                   if word2idx[word]==nil then
                      idx2word[#idx2word + 1] = word -- create word-idx/idx-word mappings
                      word2idx[word] = #idx2word
                   end
                   output_tensors[split][word_num] = word2idx[word]
                   if use_morpho then 
                      if morpho_dict[word2idx[word]] == nil then 
                         print("no morpho for:", word)
                         output_morphos[split][word_num] = torch.ones(max_factor_l)
                      else
                         output_morphos[split][word_num] = morpho_dict[word2idx[word]]
                      end

                   end
                end
              
              
                for char in word:gmatch'.' do
                   if char2idx[char]==nil then
                      idx2char[#idx2char + 1] = char -- create char-idx/idx-char mappings
                      char2idx[char] = #idx2char
                   end
                   chars[#chars + 1] = char2idx[char]
                end
                chars[#chars + 1] = char2idx[tokens.END] -- end-of-word symbol
                for i = 1, math.min(#chars, max_word_l) do
                   output_chars[split][word_num][i] = chars[i]
                end
             end
             append(rword)
          end
	  if tokens.EOS ~= '' then --PTB does not have <eos> so we add a character for <eos> tokens
              append(tokens.EOS)   --other datasets don't need this
	  end
       end
    end
    print "done"
    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, {idx2word, word2idx, idx2char, char2idx, idx2factor, factor2idx})
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensors)
    print('saving ' .. out_charfile)
    torch.save(out_charfile, output_chars)
    if use_morpho then 
       print('saving ' .. out_morphofile)
       torch.save(out_morphofile, output_morphos)
    end
end

return BatchLoaderUnk

