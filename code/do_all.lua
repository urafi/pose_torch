require 'nn'
require 'cudnn'
require 'cutorch'
require 'cunn'
----------------------------------------------------------------------
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:option('-dataset', 'flic', 'choose dataset to train on')
cmd:text()



opt = {
 
        dataset = cmd:parse(arg).dataset,
        global_step = 0,
        learningRate = .00092,
        batchSize = 8,
        threads = 1,
        seed = 1,
        total_epochs = 1
       }
 


torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

print(opt)
output_size = 256


----------------------------------------------------------------------
print '==> executing all'

dofile 'data.lua'
dofile 'model/MRFCGN.lua'
dofile 'loss.lua'
dofile 'train.lua'
dofile 'test.lua'

----------------------------------------------------------------------

-- params for learning rate decay
local params = {}
params.learning_rate = opt.learningRate
params.global_step = opt.global_step
params.decay_rate = 0.95
params.decay_after_epoch = 73
params.decay_steps = torch.round(((#trainData)/opt.batchSize) * params.decay_after_epoch)
params.staircase = 1

print '==> training and validating!'

for ep=1, opt.total_epochs do

   train()
   test()
   params.learning_rate = opt.learningRate
   params.global_step = opt.global_step
   opt.learningRate = exponential_decay(params) 

end


