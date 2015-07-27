local HighwayMLP = {}

function HighwayMLP.mlp(size, num_layers, bias, f)
    -- size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)
    
    local output, transform_gate, carry_gate
    local num_layers = num_layers or 1
    local bias = bias or -2
    local f = f or nn.ReLU()
    local input = nn.Identity()()
    local inputs = {[1]=input}
    for i = 1, num_layers do        
        output = f(nn.Linear(size, size)(inputs[i]))
        transform_gate = nn.Sigmoid()(nn.AddConstant(bias)(nn.Linear(size, size)(inputs[i])))
        carry_gate = nn.AddConstant(1)(nn.MulConstant(-1)(transform_gate))
	transform_gate = nn.CMulTable()({transform_gate, output})
	carry_gate = nn.CMulTable()({carry_gate, inputs[i]})
	output = nn.CAddTable()({transform_gate, carry_gate})
	table.insert(inputs, output)
    end
    return nn.gModule({input},{output})
end

return HighwayMLP