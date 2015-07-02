require 'nn'
require 'TensorProd'
x1 = torch.randn(40,15)
x2 = torch.randn(40,10)
y = torch.randn(40,1)
m = nn.Sequential()
m:add(nn.TensorProd(15,10,5))
m:add(nn.Linear(5,1))
crit = nn.MSECriterion()

m:zeroGradParameters()
pred = m:forward({x1,x2})
loss = crit:forward(pred, y)
dl_dp = crit:backward(pred,y)
grad_input = m:backward({x1,x2}, dl_dp)
grad_x1, grad_x2 = table.unpack(grad_input)
eps = 1e-4


for i = 1, grad_x2:size(1) do
    for j = 1, grad_x2:size(2) do
        x2[i][j] = x2[i][j] + eps
	new_pred = m:forward({x1,x2})
	new_loss = crit:forward(new_pred, y)
	grad_est = (new_loss - loss)/eps
	print("est: " .. grad_est .. ", act: " .. grad_x2[i][j])
	x2[i][j] = x2[i][j] - eps
    end
end

for i = 1, m:get(1).bias:size(1) do
    m:get(1).bias[i] = m:get(1).bias[i] + eps
    new_pred = m:forward({x1,x2})
    new_loss = crit:forward(new_pred, y)
    grad_est = (new_loss - loss)/eps
    print("est: " .. grad_est .. ", act: " .. m:get(1).gradBias[i])
    m:get(1).bias[i] = m:get(1).bias[i] - eps    
end

t = m:get(1)
for i = 1, t.weight:size(1) do
    for j = 1, t.weight:size(2) do
    	for k = 1, t.weight:size(3) do 
	    t.weight[i][j][k] = t.weight[i][j][k] + eps
	    new_pred = m:forward({x1,x2})
	    new_loss = crit:forward(new_pred, y)
	    grad_est = (new_loss - loss)/eps
	    print("est: " .. grad_est .. ", act: " .. t.gradWeight[i][j][k])
	    t.weight[i][j][k] = t.weight[i][j][k] - eps
	end
    end
end