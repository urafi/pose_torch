require 'torch'
require 'optim'
require 'xlua'
require 'image'
require 'util.lua'

--transfer model and crietrion on GPU
model:cuda()
criterion:cuda()

parameters,gradParameters = model:getParameters()

optimState = {

	learningRate = opt.learningRate,
	beta1 = 0.9,
        epsilon = 0.1
     }          

function train()

   -- epoch tracker
   epoch = epoch or 1
   optimState.learningRate = opt.learningRate
   -- local vars
   local time = sys.clock()
   local trainError = 0
   local trainAcc = torch.zeros(trainData.joints)
   model:training()
  
   -- shuffle at each epoch
   shuffle = torch.randperm(trSize)
 
   
   
   local inputsGPU = torch.CudaTensor()
   local inputsGPU2 = torch.CudaTensor()
   local targetsGPU = torch.CudaTensor()
   
   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  
   
   for t = 1,trSize,opt.batchSize do
      -- disp progress
      
      xlua.progress(t, trSize)
      local inputsCPU = torch.FloatTensor(math.min(opt.batchSize,trSize - t + 1), 3, output_size, output_size)
      local inputsCPU2 = torch.FloatTensor(math.min(opt.batchSize,trSize - t +1),3, output_size/2, output_size/2)
      local targetsCPU = torch.FloatTensor(math.min(opt.batchSize,trSize - t +1), trainData.joints , output_size, output_size)
      local scales = torch.FloatTensor(math.min(opt.batchSize,trSize - t +1))
      local state = torch.ByteTensor(math.min(opt.batchSize,trSize - t +1), trainData.joints)
      local joints_batch = torch.FloatTensor(math.min(opt.batchSize,trSize - t +1), 3, trainData.joints)
     
      opt.global_step = opt.global_step + 1
      collectgarbage() 
      
      -- create mini batch
      local idx = 1
      for i = t,math.min(t+opt.batchSize-1,trSize) do
         -- load new sample
         local input =  image.load(trainData[shuffle[i]].image)
         input:float():div(255)

         local example = {}
         example.image = input
         example.joints = trainData[shuffle[i]].joints
         if opt.dataset == 'mpi' then
                example.head_rect = trainData[shuffle[i]].head_rect
         else 
         	example.torso = trainData[shuffle[i]].torso
         end
         example.crop_pos = trainData[shuffle[i]].crop_pos
         example.is_train = 1
         example.sigma = 8
         example.max_dim = trainData[shuffle[i]].max_dim
         example.output_size = output_size 
         example.no_joints = trainData.joints
         example.is_visible = trainData[shuffle[i]].is_visible  
         example.scale = trainData[shuffle[i]].scale
         example.test_mode = 0
         example_warped = apply_augmentation(example)
         
         local py = image.gaussianpyramid(example_warped.image,{1,0.5})
         inputsCPU[idx] =  py[1]
         inputsCPU2[idx] = py[2]

         targetsCPU[idx] = example_warped.gt_maps
         joints_batch[idx] = example_warped.joints 
         scales[idx] = example_warped.scale
         state[idx] =  example_warped.is_visible
         idx = idx + 1
 
      end
      
      inputsGPU:resize(inputsCPU:size()):copy(inputsCPU)
      inputsGPU2:resize(inputsCPU2:size()):copy(inputsCPU2)
      targetsGPU:resize(targetsCPU:size()):copy(targetsCPU)
      
      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()
                       
                          -- estimate f
                         local  output = model:forward{inputsGPU, inputsGPU2}
                         local N = output:size(1)

                          local err = criterion:forward(output, targetsGPU)
                       
                          -- estimate df/dW
                          local df_do = criterion:backward(output, targetsGPU)
                          for im = 1 , N do
                             for j = 1 , trainData.joints do
                                 if state[im][j] == 0 then
                                    df_do[im][j]:zero()
                                 end
                             end
                          end    
                                
                          model:backward({inputsGPU, inputsGPU2}, df_do)
                           
                          for im = 1, N do
                              local correct_predictions = torch.zeros(trainData.joints)
                              if opt.dataset == 'mpi' then
                                      correct_predictions = compute_pckh(output[im],joints_batch[im],scales[im],50)
                              else  
                                      correct_predictions = compute_pck(output[im],joints_batch[im],scales[im],20)
                              end		         
                              trainAcc:add(correct_predictions)			  
                          end			         
                              
                       
                          return f,gradParameters
                    end

        
        optim.adam(feval, parameters,optimState)
     
   end
  
   trainError = trainError / math.floor(trSize/opt.batchSize)
  
   trainAcc = trainAcc:div(trSize):mul(100)
   local over_all = trainAcc:sum()/trainData.joints
   -- time taken
   time = sys.clock() - time
   time = time 
   print("\n==> time for 1 epoch = " .. (time) .. 's')
   print("==> PCK for Joints(Training) ==> \n")
   for j = 1 , trainData.joints do
       print (trainData.joint_names[j] .. ' : ' .. string.format('%.2f',trainAcc[j]) .. '%')
   end
   print("\n == OverAll PCK(Training) == \n")
   print(string.format('%.2f',over_all) .. '%\n')
   collectgarbage()
   epoch = epoch + 1

end

