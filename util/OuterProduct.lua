local OuterProduct, parent = torch.class('nn.OuterProduct', 'nn.Module')

function OuterProduct:__init()
    parent.__init(self)
    self.gradInput = {torch.Tensor(), torch.Tensor()}
end

function OuterProduct:updateOutput(input)
    assert(#input==2, 'only supports outer products of 2 vectors')
    local a, b = table.unpack(input)
    assert(a:nDimension() == 1 or a:nDimension() == 2, 'input tensors must be 1D or 2D')
    if a:nDimension()==1 then 
       	self.output:resize(a:size(1), b:size(1))
	self.output:ger(a, b)
    else -- mini batch processing
        self.output:resize(a:size(1), a:size(2), b:size(2))
	for i = 1, a:size(1) do
	    self.output[i]:ger(a[i], b[i])
	end
    end
    return self.output
end

function OuterProduct:updateGradInput(input, gradOutput)
    local a, b = table.unpack(input)
    self.gradInput[1]:resizeAs(a)
    self.gradInput[2]:resizeAs(b)
    if a:nDimension()==1 then
        self.gradInput[1]:mv(gradOutput, b)
	self.gradInput[2]:mv(gradOutput:t(), a)
    else -- mini batch processing
        for i = 1, gradOutput:size(1) do
	    self.gradInput[1][i]:mv(gradOutput[i], b[i])
	    self.gradInput[2][i]:mv(gradOutput[i]:t(), a[i])
	end
    end
    return self.gradInput
end