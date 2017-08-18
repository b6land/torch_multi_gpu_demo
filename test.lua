require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'hdf5'

myFile = hdf5.open('/data/test.hdf5', 'r')
testset = myFile:read():all()
testset.xs = testset.xs:double()
testset.ys = torch.reshape(testset.ys, testset.ys:size(1))

net = torch.load('model.t7')
testset.xs = testset.xs:cuda()
testset.ys = testset.ys:cuda()

correct = 0
for i = 1, testset.xs:size(1) do
    local groundtruth = testset.ys[i]
    local prediction = net:forward(testset.xs[i])
    local confidences, indices = torch.sort(prediction, true)
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100 * correct / testset.xs:size(1) .. '%')

