--[[
 Time-delayed Neural Network (i.e. 1-d CNN) with multiple filter widths
--]]
local TDNN = {}

function TDNN.tdnn(length, input_size, feature_maps, kernels)
    -- length = length of sentences/words (zero padded to be of same length)
    -- input_size = embedding_size
    -- feature_maps = table of feature maps (for each kernel width)
    -- kernels = table of kernel widths
    local layer1_concat, output
    local input = nn.Identity()() --input is batch_size x length x input_size
    local layer1 = {}
    for i = 1, #kernels do
	local reduced_l = length - kernels[i] + 1 
	local conv = nn.TemporalConvolution(input_size, feature_maps[i], kernels[i])
	conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
	local conv_layer = conv(input)
	local pool_layer = nn.TemporalMaxPooling(reduced_l)(nn.Tanh()(conv_layer))
	table.insert(layer1, pool_layer)
    end
    if #kernels > 1 then
	layer1_concat = nn.JoinTable(3)(layer1)
	output = nn.Squeeze()(layer1_concat)
    else
        output = nn.Squeeze()(layer1[1])    
    end
    return nn.gModule({input}, {output})
end

return TDNN