require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'

net = nn.Sequential()
net:add(nn.SpatialConvolution(16, 92, 5, 5, 1, 1, 2, 2))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(92, 384, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1))
net:add(nn.ReLU())
net:add(nn.SpatialConvolution(384, 1, 3, 3, 1, 1, 1, 1))
net:add(nn.View(1*361))
net:add(nn.LogSoftMax())
net = net:cuda()
print(net)

torch.save('model.t7', net)
