local Diag, parent = torch.class('nn.Diag', 'nn.Module')

function Diag:__init(pos, length)
    parent.__init(self)
    self.weight = torch.Tensor(pos, length)
    self.gradWeight = torch.Tensor(pos, length)
    self:reset()
end

function Diag:reset(stdv)
    stdv = stdv or 1./math.sqrt(self.weight:size(1) + self.weight:size(2))
    self.weight:uniform(-stdv, stdv) 
end

function Diag:updateOutput(input)
    if input:dim()==2 then
        self.output:resize(self.weight:size(1), self.weight:size(2))
	self.output:cmul(input, self.weight)
    else
        local batch_size = input:size(1)
	if not self.tmp or self.tmp:size(1) ~= batch_size then
	    self.tmp = torch.expand(self.weight:view(1,self.weight:size(1), 
	    	     self.weight:size(2)), batch_size, self.weight:size(1), self.weight:size(2))
	end
	self.output:resize(batch_size, self.weight:size(1), self.weight:size(2))
	self.output:cmul(input, self.tmp)	
    end
    return self.output
end

function Diag:updateGradInput(input, gradOutput)
    if input:dim()==2 then
        self.gradInput:resize(self.weight:size(1), self.weight:size(2))
	self.gradInput:cmul(gradOutput, self.weight)
    else
        local batch_size = input:size(1)
	self.gradInput:resize(batch_size, self.weight:size(1), self.weight:size(2))
	self.gradInput:cmul(gradOutput, self.tmp)
    end
    return self.gradInput
end

function Diag:accGradParameters(input, gradOutput)
    if input:dim()==2 then
        self.gradWeight:addcmul(gradOutput, input)
    else
        local batch_size = input:size(1)
        if not self.tmpGrad or self.tmpGrad:size(1) ~= batch_size then
	    self.tmpGrad = torch.Tensor(batch_size, self.weight:size(1),
	       		   			    self.weight:size(2))
	end
	self.tmpGrad:cmul(gradOutput, input)
        self.gradWeight:add(self.tmpGrad:sum(1):squeeze())
    end
end