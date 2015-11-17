-- Time-delayed Neural Network (i.e. 1-d CNN) with multiple filter widths

local TDNN = {}
--local cudnn_status, cudnn = pcall(require, 'cudnn')

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
       local pool_layer
       if opt.cudnn == 1 then
          -- Use CuDNN for temporal convolution.
          if not cudnn then require 'cudnn' end
          -- Fake the spatial convolution.
          local conv = cudnn.SpatialConvolution(1, feature_maps[i], input_size,
                                                kernels[i], 1, 1, 0)
          conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
          local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
          --pool_layer = nn.Max(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
	  pool_layer = nn.Squeeze()(cudnn.SpatialMaxPooling(1, reduced_l, 1, 1, 0, 0)(nn.Tanh()(conv_layer)))
       else
          -- Temporal conv. much slower
          local conv = nn.TemporalConvolution(input_size, feature_maps[i], kernels[i])
          local conv_layer = conv(input)
          conv.name = 'conv_filter_' .. kernels[i] .. '_' .. feature_maps[i]
          --pool_layer = nn.Max(2)(nn.Tanh()(conv_layer))    
	  pool_layer = nn.TemporalMaxPooling(reduced_l)(nn.Tanh()(conv_layer))
	  pool_layer = nn.Squeeze()(pool_layer)

       end
       table.insert(layer1, pool_layer)
    end
    if #kernels > 1 then
       layer1_concat = nn.JoinTable(2)(layer1)
       output = layer1_concat
    else
        output = layer1[1]
    end
    return nn.gModule({input}, {output})
end

return TDNN
