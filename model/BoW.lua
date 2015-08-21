
local BoW = {}

function BoW.bow(length, input_size)
    local input = nn.Identity()() --input is batch_size x length x input_size
    local output = nn.Sum(2)(input) --nn.Sum(3)(nn.Transpose({2, 3})(input))
    return nn.gModule({input}, {output})
end

return BoW
