local TensorProd, parent = torch.class('nn.TensorProd', 'nn.Module')

function TensorProd:__init(vec1_size, vec2_size, output_size)
    parent.__init(self)
    self.bias = torch.Tensor(output_size)
    self.gradBias = torch.Tensor(output_size)
    self.weight = torch.Tensor(output_size, vec1_size, vec2_size)
    self.gradWeight = torch.Tensor(output_size, vec1_size, vec2_size)
    --self.tmp = torch.Tensor()
    self.gradInput = {torch.Tensor(), torch.Tensor()}
    --self.ab = torch.Tensor(vec1_size, vec2_size) -- outer prods 
    self:reset()
end

function TensorProd:reset(stdv)
    stdv = sdtv or 1./math.sqrt(self.weight:size(2) + self.weight:size(3))
    self.weight:uniform(-stdv, stdv)
    self.bias:uniform(-stdv, stdv)
end

function TensorProd:updateOutput(input)
    local a, b = table.unpack(input) -- output is a x weight x b
    assert(a:dim()==b:dim(), 'input tensors should have same number of dims (1 or 2)')
    if a:dim()==1 then
        self.output:resize(self.weight:size(1))
	--self.tmp:resize(self.weight:size(1), self.weight:size(2))
	for i = 1, self.weight:size(1) do
	    --self.tmp[i]:mv(self.weight[i], b)	    
	    --self.output[i] = self.tmp[i]:dot(a)
	    self.output[i] = a:dot(torch.mv(self.weight[i],b))
	end
	self.output = self.output + self.bias
    elseif a:dim()==2 then -- mini-batch processing
        local batch_size = a:size(1)
	self.output:resize(batch_size, self.weight:size(1))
	--self.tmp:resize(self.weight:size(1), batch_size, self.weight:size(2))
	for i = 1, self.weight:size(1) do
	    --self.tmp[i]:mm(b, self.weight[i]:t())	    
	    self.output[{{},i}] = torch.sum(torch.cmul(torch.mm(b, self.weight[i]:t()), a),2) + self.bias[i]
	end
    else
        error("input must be 1D or 2D tensors")
    end
    return self.output
end

function TensorProd:updateGradInput(input, gradOutput)
    local a, b = table.unpack(input)
    self.gradInput[1]:resizeAs(a)
    self.gradInput[2]:resizeAs(b)
    if a:dim() == 1 then
	for i = 1, self.weight:size(1) do
	    self.gradInput[1]:addmv(gradOutput[i], self.weight[i], b)
	    self.gradInput[2]:addmv(gradOutput[i], self.weight[i]:t(), a)
	end
    else -- mini-batch processing
	local gradOutput1 = gradOutput:view(a:size(1), self.weight:size(1),
	    				1):expand(a:size(1), self.weight:size(1), self.weight:size(2))
	local gradOutput2 = gradOutput:view(a:size(1), self.weight:size(1),
	    				1):expand(a:size(1), self.weight:size(1), self.weight:size(3))					
	for i = 1, self.weight:size(1) do 
	    self.gradInput[1]:add(torch.cmul(torch.mm(b, self.weight[i]:t()), gradOutput1[{{},i}]))
	    self.gradInput[2]:add(torch.cmul(torch.mm(a, self.weight[i]), gradOutput2[{{},i}]))
	end
    end
    return self.gradInput
end

function TensorProd:accGradParameters(input, gradOutput)
    local a, b = table.unpack(input)
    if a:dim()==1 then
        local ab = torch.ger(a,b)
	self.gradBias:add(gradOutput)
	for i = 1, self.weight:size(1) do
	    self.gradWeight[i]:add(ab*gradOutput[i])
	end    
    else -- mini-batch processing
        self.gradBias:add(gradOutput:sum(1))
	for i = 1, a:size(1) do
	    local ab = torch.ger(a[i],b[i])
	    for j = 1, self.weight:size(1) do
	        self.gradWeight[j]:add(ab*gradOutput[i][j])
	    end
	end
    end
end

function TensorProd:__tostring__()
    return torch.type(self) ..
        string.format('(%d x %d -> %d)', self.weight:size(2), self.weight:size(3), self.weight:size(1))
end

