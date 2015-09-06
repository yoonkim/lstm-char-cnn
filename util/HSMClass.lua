local HSMClass = {}

function HSMClass.hsm(input_size, n_clusters, n_max_class_in_cluster)
    --inputs[1] is the input (batch_size by input_size)
    --inputs[2] is the target cluster (batch_size)

    local inputs = {nn.Identity()(), nn.Identity()()}
    local class_bias_layer = nn.LookupTable(n_clusters, n_max_class_in_cluster)
    local class_vec_layer = nn.LookupTable(n_clusters, input_size*n_max_class_in_cluster)
    local class_mat = nn.View(n_max_class_in_cluster, input_size)(
                                class_vec_layer(inputs[2]))
    class_bias_layer.name = 'class_bias'
    class_vec_layer.name = 'class_weight'
    local input_mat = nn.View(input_size, 1)(inputs[1])
    local class_scores = nn.Squeeze()(nn.MM()({class_mat, input_mat}))
    local output = nn.LogSoftMax()(nn.CAddTable()({class_bias_layer(inputs[2]), class_scores}))
    return nn.gModule(inputs, {output})
end

return HSMClass