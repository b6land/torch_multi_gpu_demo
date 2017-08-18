-- reference : https://github.com/torch/cunn/blob/master/test_DataParallelTable.lua
-- must check multi-gpu is training in the same time or not

require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'hdf5'
require 'optim'
math.randomseed(os.time())

function print_msg(msg)
    print(os.date("[%b %d %H:%M:%S] ") .. msg)
end

function load_file()
    -- hardcode path
    local path = '/data/data.hdf5'

    local myFile = hdf5.open(path, 'r')
    trainset = myFile:read():all()
    trainset.xs = trainset.xs:double()
    trainset.ys = trainset.ys:double()
end

local maxEpoch = 20
local baseGpu = 1 
local numGpus = 2
torch.setdefaulttensortype('torch.DoubleTensor') 
cutorch.setDevice(baseGpu)

for epoch = 1, maxEpoch do

    net = torch.load('model.t7')

    -- Build a multi-GPU model 
    local gNet = nn.DataParallelTable(1):threads(function()
        require 'cudnn'
    end)  

    for i = 1, numGpus do
        local curGpu = math.fmod(baseGpu+(i-1)-1, cutorch.getDeviceCount()) + 1
        cutorch.setDevice(curGpu)
        gNet:add(net:clone(), curGpu)
    end
    cutorch.setDevice(baseGpu)

    local criterion = nn.ClassNLLCriterion():cuda()
    local params, gradParams = gNet:getParameters()

    assert(cutorch.getDevice() == baseGpu,'getParameters: didnt restore GPU state')  

    -- training options    
    local optimState = {learningRate = 0.01}
    local batchSize = 256
    local batchTimes = 500

    print_msg(' (' .. epoch .. ') Training...')
    for t_m = 1, 50 do
        load_file()
        shuffle = torch.randperm(trainset.xs:size(1))
    
        -- training batches
        for m = 1, batchTimes * batchSize, batchSize do
        
            -- careful when data size less than (batch size * batch times)
            local batchInputs = torch.Tensor(batchSize, trainset.xs:size(2), trainset.xs:size(3), trainset.xs:size(4))
            local batchLabels = torch.Tensor(batchSize)
            for i = 1, batchSize do
                batchInputs[{{i}, {}, {}, {}}] = trainset.xs[{{shuffle[m + i - 1]}, {}, {}, {}}]
                batchLabels[{{i}}] = trainset.ys[{{shuffle[m + i - 1]}}]
            end
            batchInputs = batchInputs:cuda()
            batchLabels = batchLabels:cuda()
    
            local feval = function(x)
                if x ~= params then params:copy(x) end 
                gNet:zeroGradParameters() 
    
                assert(cutorch.getDevice() == baseGpu, 'zeroGradParameters: didnt restore GPU state')
                local outputs = gNet:forward(batchInputs)
                assert(cutorch.getDevice() == baseGpu, 'DataParallelTable:forward didnt restore GPU state')
                local loss = criterion:forward(outputs, batchLabels)
                local dloss_output = criterion:backward(outputs, batchLabels)
                local dloss_input = gNet:backward(batchInputs, dloss_output)
                assert(cutorch.getDevice() == baseGpu, 'DataParallelTable:add didnt restore GPU state')
    
                return loss, gradParams
            end
            optim.sgd(feval, params, optimState)
            gNet:findModules('nn.DataParallelTable')[1]:syncParameters() 
            assert(cutorch.getDevice() == baseGpu, 'DataParallelTable:syncParameters didnt restore GPU state')
        end
    
    end
    
    -- training process finish
    -- hardcode path    
    print_msg('Finished, save model.')
    clearNet = gNet:get(1):clearState()
    torch.save('model.t7', clearNet)
    collectgarbage()
end
