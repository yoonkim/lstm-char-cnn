local LookupTableInt, parent = torch.class('nn.LookupTableInt', 'nn.Module')

function LookupTableInt:__init(init_lookup)
  parent.__init(self)
  self.weight = init_lookup:clone():long()
  self.size = {init_lookup:size(1), init_lookup:size(2)}
  self.output = torch.LongTensor()
end

function LookupTableInt:updateOutput(input)
  -- make sure input is a contiguous torch.LongTensor
  if (not input:isContiguous()) or torch.type(input) ~= 'torch.LongTensor' then
      self._indices = self._indices or torch.LongTensor()
      self._indices:resize(input:size()):copy(input)
      input = self._indices
  end
  if input:dim() == 1 then
      local nIndex = input:size(1)
      self.output:index(self.weight, 1, input)
  elseif input:dim() == 2 then
      -- batch mode
      local nExample = input:size(1)
      local nIndex = input:size(2)
      self._inputView = self._inputView or torch.LongTensor()
      self._inputView:view(input, -1)
      self.output:index(self.weight, 1, self._inputView)
      self.output = self.output:view(nExample, nIndex, self.size[2])
  end
  return self.output
end