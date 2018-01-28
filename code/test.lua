require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'
require 'util.lua'

----------------------------------------------------------------------
print '==> defining test procedure'
local AccLogger = optim.Logger(paths.concat('progress/testAcc_' .. opt.dataset .. '.log'))
local max_acc = 0
local im_counter = 1
-- test function
function test()
   -- local vars
   local time = sys.clock()

  
   local counter = 1
   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()
   
   local testAcc = torch.zeros(testData.joints)
   --local testErr = 0
   local inputsCPU = torch.FloatTensor(opt.batchSize,3,output_size,output_size)
   local inputsCPU2 = torch.FloatTensor(opt.batchSize,3, output_size/2, output_size/2)
   local joints_batch = torch.FloatTensor(opt.batchSize,3,testData.joints)
   local scales = torch.FloatTensor(opt.batchSize)
   local inputsGPU = torch.CudaTensor()
   local inputsGPU2 = torch.CudaTensor()
     
   print('==> testing on test set:')
   for t = 1,tsSize,opt.batchSize do
      
      -- disp progress
      xlua.progress(t, tsSize)
      collectgarbage()
      local idx = 1
      for i = t,math.min(t+opt.batchSize-1,tsSize) do
         -- load new sample
         local input = image.load(testData[i].image)
         input:float():div(255)
         
         local example = {}
         example.image = input
         example.joints = testData[i].joints
         if opt.dataset == 'mpi' then
                example.head_rect = testData[i].head_rect
         else 
         	example.torso = testData[i].torso
         end
         example.crop_pos = testData[i].crop_pos
         example.is_train = 0
         example.max_dim = testData[i].max_dim
         example.output_size = output_size 
         example.no_joints = testData.joints 
         example.scale = testData[i].scale
         example.test_mode = 0
         example_warped = apply_augmentation(example)
        
         
         local py = image.gaussianpyramid(example_warped.image,{1,0.5})
         inputsCPU[idx] =  py[1]
         inputsCPU2[idx] = py[2]
         
         joints_batch[idx] = example_warped.joints 
         scales[idx] = example_warped.scale
         idx = idx + 1 
          
      end
      
      
     inputsGPU:resize(inputsCPU:size()):copy(inputsCPU)
     inputsGPU2:resize(inputsCPU2:size()):copy(inputsCPU2)
    
     	local preds = model:forward{inputsGPU, inputsGPU2}
         
        for im = 1,opt.batchSize do
             local correct_predictions = torch.zeros(testData.joints)                 
             if opt.dataset == 'mpi' then
                correct_predictions = compute_pckh(preds[im],joints_batch[im],scales[im],50)
             else  
                correct_predictions = compute_pck(preds[im],joints_batch[im],scales[im],20)
             end
             testAcc:add(correct_predictions)			  
        end	
end
      -- test sample
   collectgarbage()  
   
   -- timing
   testAcc = testAcc:div(tsSize):mul(100)
   local over_all = testAcc:sum()/testData.joints
   time = sys.clock() - time
   time = time 
   print("\n==> time to test 1 epoch = " .. (time) .. 's')
   print("==> PCK for Joints(Test) ==> \n")
   for j = 1 , testData.joints do
       print (testData.joint_names[j] .. ' : ' .. string.format('%.2f',testAcc[j]) .. '%')
   end
   print("\n == OverAll PCK(Test) == \n")
   print(string.format('%.2f',over_all) .. '%\n')

   -- plot the accuracy graph
   AccLogger:add{['% Acc (test set)'] = over_all}
   AccLogger:plot()

   --save the best scoring model so far
   if over_all > max_acc then
      max_acc = over_all
      torch.save(paths.concat('progress/MRFCGN_' .. opt.dataset .. '.t7'), model)
   end

end

