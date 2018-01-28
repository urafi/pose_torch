require 'image'
require 'nn'
require 'cudnn'
require 'cunn'
require 'util.lua'

local matio = require 'matio'

-- function to rewarp joint coords to the orignal image resolution ---  
function re_warp(preds, warp)

    local N = preds:size(1)
    local joints_rewarped = torch.FloatTensor(N, testData.joints, 2)
    for i = 1, N do	
        for j = 1 , testData.joints do
   
	local vals,mrows = torch.max(preds[i][j],2)
   	local val,row = torch.max(vals,1)
   
        local col = mrows[row[1][1]]
        local y_pred = row[1][1]
        local x_pred = col[1]
        local joint = torch.FloatTensor(3,1)
        joint[1][1] = x_pred
        joint[2][1] = y_pred
        joint[3][1] = 1
        local joint_rewarped = torch.mm(warp[i],joint) 
        joints_rewarped[i][j][1] = joint_rewarped[1] 
        joints_rewarped[i][j][2] = joint_rewarped[2] 
        
        end
    end

return joints_rewarped

end



local batchSize = 8
torch.setdefaulttensortype('torch.FloatTensor')
local output_size = 256 -- warp size for image

cmd = torch.CmdLine()
cmd:option('-dataset', 'flic', 'choose dataset to evaluate')

opt = {}
opt.dataset = cmd:parse(arg).dataset


print ("Evaluating on " .. opt.dataset)

-- load the model and set it to evaluation
local model = torch.load('pretrained_models/MRFCGN_' .. opt.dataset .. '.t7') 
model = model:cuda() 
model:evaluate() 


--load the dataset
testData =  torch.load('data/test_set_' .. opt.dataset .. '.t7')
local tsSize = #testData

--tensors for full and half res images
local inputsGPU = torch.CudaTensor()
local inputsGPU2 = torch.CudaTensor()


--tensor for predictions
local predictions = torch.FloatTensor(tsSize, testData.joints, 2)

-- loop thorugh the testset 
for t = 1, tsSize, batchSize do

      xlua.progress(t, tsSize)

      local inputsCPU = torch.FloatTensor(math.min(batchSize,tsSize - t + 1),3,output_size,output_size)
      local inputsCPU2 = torch.FloatTensor(math.min(batchSize,tsSize - t +1),3, output_size/2, output_size/2)

      collectgarbage()
      local idx = 1
      local warps_batch = {}
      for i = t,math.min(t + batchSize - 1, tsSize) do
      
         -- load image and warp it to 256 x 256  
         local input = image.load(testData[i].image)
         input:float():div(255)
         local example = {}
         example.image = input
         example.testmode = 1
         example.crop_pos = testData[i].crop_pos
         example.max_dim = testData[i].max_dim
         example.is_train = 0
         example.output_size = output_size
         example.returnwarp = 1 
         example.scale = testData[i].scale
         example.no_joints =testData.joints
         example_warped = apply_augmentation(example)
         local py = image.gaussianpyramid(example_warped.image,{1,0.5}) -- creates full and half res image
         inputsCPU[idx] =  py[1]
         inputsCPU2[idx] = py[2]
         table.insert(warps_batch,example_warped.warp_inv) 
         idx = idx + 1 
         
       
          
      end

      
      
      inputsGPU:resize(inputsCPU:size()):copy(inputsCPU)
      inputsGPU2:resize(inputsCPU2: size()):copy(inputsCPU2)
      local preds = model:forward{inputsGPU,inputsGPU2}
      rewarped_preds = re_warp(preds, warps_batch) -- rewarp joint coords to orignal image resolution
      idx = 1
      for i = t , math.min(t + batchSize - 1, tsSize) do
          predictions[i] = rewarped_preds[idx]
          idx = idx + 1
      end  
      
 end

matio.save('predictions/predictions_' .. opt.dataset .. '.mat', predictions)
