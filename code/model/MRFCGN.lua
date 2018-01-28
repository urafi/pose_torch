require 'torch'
require 'nn'
require 'cudnn'

-- Our adaptation of BN Google network

-- The code for our model is inspired by the code from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/googlenet.lua. Thanks to Soumith Chintala.

function inception_module(depth_dim, input_size, config)
  
   local conv1 = nil   
   local conv3 = nil
   local conv3x3 = nil
   local pool = nil
   local depth_concat = nn.DepthConcat(depth_dim)
   --conv1
   
   if config[1][1] ~= 0 then
     conv1 = nn.Sequential()
     conv1:add(cudnn.SpatialConvolution(input_size, config[1][1], 1, 1))
     conv1:add(nn.SpatialBatchNormalization(config[1][1],1e-3))
     conv1:add(nn.ELU())
     depth_concat:add(conv1)
   end
   --conv3
   conv3 = nn.Sequential()
   conv3:add(cudnn.SpatialConvolution(input_size, config[2][1], 1, 1))
   conv3:add(nn.SpatialBatchNormalization(config[2][1],1e-3))
   conv3:add(nn.ELU())
   conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2], 3, 3,config[2][3],config[2][3],1,1))
   conv3:add(nn.SpatialBatchNormalization(config[2][2],1e-3))
   conv3:add(nn.ELU())
   depth_concat:add(conv3)
  
   --conv3x3
   conv3x3 = nn.Sequential()
   conv3x3:add(cudnn.SpatialConvolution(input_size, config[3][1], 1, 1))
   conv3x3:add(nn.SpatialBatchNormalization(config[3][1],1e-3))
   conv3x3:add(nn.ELU())
   conv3x3:add(cudnn.SpatialConvolution(config[3][1], config[3][2], 3, 3,1,1,1,1))
   conv3x3:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
   conv3x3:add(nn.ELU())
   conv3x3:add(cudnn.SpatialConvolution(config[3][2], config[3][2], 3, 3,config[3][3],config[3][3],1,1))
   conv3x3:add(nn.SpatialBatchNormalization(config[3][2],1e-3))
   conv3x3:add(nn.ELU())
   depth_concat:add(conv3x3)
  
   --pool 
   pool = nn.Sequential()
   if config[4][1] == 'max' then
      pool:add(cudnn.SpatialMaxPooling(3,3,config[4][3],config[4][3]):ceil())
   elseif config[4][1] == 'avg' then
      pool:add(cudnn.SpatialAveragePooling(3,3,config[4][3],config[4][3]):ceil())
   else
      error('Unknown pooling')
   end
   if config[4][2] ~= 0  then
        pool:add(cudnn.SpatialConvolution(input_size, config[4][2], 1, 1))
        pool:add(nn.SpatialBatchNormalization(config[4][2],1e-3))
        pool:add(nn.ELU())
   end
   depth_concat:add(pool)
   
   return depth_concat

end


--- Fully Convolutional Google Net-----   
  
   local f = nn.Sequential()
   f:add(cudnn.SpatialConvolution(3, 64, 7, 7, 2, 2, 3,3))
   f:add(nn.SpatialBatchNormalization(64,1e-3))
   f:add(nn.ELU())
   f:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
   f:add(cudnn.SpatialConvolution(64,64,1,1))
   f:add(nn.SpatialBatchNormalization(64,1e-3))
   f:add(nn.ELU())
   f:add(cudnn.SpatialConvolution(64,192,3,3,1,1,1,1))
   f:add(nn.SpatialBatchNormalization(192,1e-3))
   f:add(nn.ELU())
   f:add(nn.SpatialMaxPooling(3,3,2,2):ceil())

   -- inception 3a
   f:add(inception_module(2, 192, {{64}, {64, 64,1}, {64,96,1}, {'avg', 32,1}}))

   -- inception 3b
   f:add(inception_module(2, 256, {{64}, {64, 96,1}, {64, 96,1}, {'avg', 64,1}}))
   
   -- inception 3c
   f:add(inception_module(2, 320, {{0}, {128, 160,1}, {64, 96,1}, {'max', 0,1}}))

   f:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())

   -- inception 4a
   f:add(inception_module(2, 576, {{224}, {64, 96,1}, {96, 128,1}, {'avg', 128,1}}))

   -- inception 4b
   f:add(inception_module(2, 576, {{192}, {96, 128,1}, {96, 128,1}, {'avg', 128,1}}))

   -- inception 4c
   f:add(inception_module(2, 576, {{160}, {128, 160,1}, {128, 160,1}, {'avg', 96,1}}))

   -- inception 4d
   f:add(inception_module(2, 576, {{96}, {128, 192,1}, {160, 192,1}, {'avg', 96,1}}))

   local main = nn.Sequential() 

   -- inception 4e
   main:add(inception_module(2, 576, {{0}, {128, 192,1}, {192, 256,1}, {'max', 0,1}}))

   main:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())


   -- inception 5a
   main:add(inception_module(2, 1024, {{352}, {192, 320}, {160, 224}, {'avg', 128,1}}))

   -- inception 5b
   main:add(inception_module(2, 1024, {{352}, {192, 320}, {192, 224}, {'max', 128,1}}))

   main:add(nn.SpatialFullConvolution(1024,576,2,2,2,2))   
   main:add(nn.SpatialBatchNormalization(576,1e-3))
   main:add(nn.ELU())
   
   local splitter = nn.Concat(2)
   splitter:add(nn.Identity()) 
   splitter:add(main)


--- Main MultiResolution Network ----

   local FCGN1 = nn.Sequential()

   FCGN1:add(f)
   FCGN1:add(splitter)


   local FCGN2= FCGN1:clone('weight','bias','gradWeight','gradBias')
   FCGN2:add(nn.SpatialFullConvolution(576*2,576*2,2,2,2,2))   
   FCGN2:add(nn.SpatialBatchNormalization(576*2,1e-3))
   FCGN2:add(nn.ELU())
   
   
   local multinet = nn.ParallelTable()
   multinet:add(FCGN1)
   multinet:add(FCGN2)


   model = nn.Sequential()
   model:add(multinet)
   model:add(nn.JoinTable(2,4))

   model:add(nn.SpatialDropout(0.4))
   model:add(nn.SpatialFullConvolution(576*4,trainData.joints,32,32,16,16,8,8))
   model:add(nn.Sigmoid())
   
