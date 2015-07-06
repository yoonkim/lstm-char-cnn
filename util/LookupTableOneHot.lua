local LookupTableOneHot, parent = torch.class('nn.LookupTableOneHot', 'nn.Module')

function LookupTableOneHot:__init(input_size)
  parent.__init(self)
  self.eye = torch.eye(input_size)
  self.output = torch.Tensor()
end

function LookupTableOneHot:updateOutput(input)
  -- make sure input is a contiguous torch.LongTensor
  if (not input:isContiguous()) or torch.type(input) ~= 'torch.LongTensor' then
      self._indices = self._indices or torch.LongTensor()
      self._indices:resize(input:size()):copy(input)
      input = self._indices
  end
  if input:dim() == 1 then
      local nIndex = input:size(1)
      self.output:index(self.eye, 1, input)
  elseif input:dim() == 2 then
      -- batch mode
      local nExample = input:size(1)
      local nIndex = input:size(2)
      self._inputView = self._inputView or torch.LongTensor()
      self._inputView:view(input, -1)
      self.output:index(self.eye, 1, self._inputView)
      self.output = self.output:view(nExample, nIndex, self.eye:size(1))
  end
  return self.output
end