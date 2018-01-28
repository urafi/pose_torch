

function generate_heatmap(joints, example)

local size =  example.output_size
local heatmaps = torch.FloatTensor(example.no_joints, size , size)
local grid_y = torch.ger( torch.linspace(1, size, size), torch.ones(size)) 
local grid_x = torch.ger( torch.ones(size), torch.linspace(1, size, size))


for i = 1 , example.no_joints do
         
         diff_x = torch.add(grid_x,-joints[1][i])
         diff_y = torch.add(grid_y,-joints[2][i])
         diff_x = diff_x:cmul(diff_x)
         diff_y = diff_y:cmul(diff_y)
         diff = torch.add(diff_x,diff_y)
         diff = torch.le(diff, (example.sigma * example.sigma))
         h = diff
         heatmaps[i] = h
 
end

return heatmaps 

end

function compute_warp_matrix(example)
	
        local valid_warp = 0
        
	local trans_to_origin = torch.diag(torch.ones(3))
	trans_to_origin[1][3] = example.output_size/2
	trans_to_origin[2][3] = example.output_size/2
        local is_flipped = 0
        local rotation = 0
        local scale = 0
	local flip = 0
        local trans_x = 0
        local trans_y = 0
        local a = 0
	local b = 0
        local affine_warp = torch.diag(torch.ones(3))
        local joints = torch.FloatTensor(3,example.no_joints)
        joints[3] = torch.ones(1,example.no_joints)
        local is_visible = torch.ByteTensor(example.no_joints)    
        while valid_warp == 0 do
        
                
                valid_warp = 1
                if example.is_train == 1 then

			rotation = torch.uniform(-20 * 0.0174532, 20 * 0.0174532)
                        if opt.dataset == 'lsp' then
                        	scale = (example.output_size / example.max_dim)  * torch.uniform(0.5,1.5)
                        elseif opt.dataset == 'flic' then
                                scale = (example.output_size / example.max_dim)  * (200/example.scale) * torch.uniform(0.5,1.5)
                        else 
                                scale = example.scale * torch.uniform(0.5,1.5)
                        end  
			flip = torch.bernoulli(0.5)
		        trans_x = torch.random(-20, 20)
		        trans_y = torch.random(-20, 20)
		        a = scale * math.cos(rotation)
			b = scale * math.sin(rotation)
		        
                        affine_warp[1][1] = a 
		        
                        if flip == 1 then
		           affine_warp[1][1] = -a 
		           is_flipped = 1
			else
			   is_flipped = 0
			end
                      
                        
		        affine_warp[1][2] = -b
		        affine_warp[2][1] = b
			affine_warp[2][2] = a
		        local trans_to_center = torch.diag(torch.ones(3))
                        trans_to_center[1][3] = -example.crop_pos[1][1] + trans_x
	                trans_to_center[2][3] = -example.crop_pos[1][2] + trans_y
                        
			warp = torch.mm(trans_to_origin,affine_warp)
			warp = torch.mm(warp,trans_to_center)
                        joints_warped = torch.mm(warp, example.joints)
                         
                        for j = 1 , example.no_joints do
		            if (example.is_visible[1][j] == 1) and (joints_warped[1][j] > output_size or joints_warped[2][j] > output_size or joints_warped[1][j] < 1 or joints_warped[2][j] < 1 ) then
		               valid_warp = 0
                            end  
		        end
                 else
                       if opt.dataset == 'lsp' then
                        	scale = (example.output_size / example.max_dim)
                       elseif opt.dataset == 'flic' then
                                scale = (example.output_size / example.max_dim)  * (200/example.scale) 
                       else 
                                scale = example.scale 
                       end  
		      
                       affine_warp[1][1] = scale
                       affine_warp[2][2] = scale 
                       warp = torch.mm(trans_to_origin,affine_warp)
                       local trans_to_center = torch.diag(torch.ones(3))
                       trans_to_center[1][3] = -example.crop_pos[1][1]
	               trans_to_center[2][3] = -example.crop_pos[1][2]
		       warp = torch.mm(warp,trans_to_center) 
                       if example.test_mode == 0 then
                       joints_warped = torch.mm(warp, example.joints)
                       end 
                       is_flipped = 0
                 end  
	end
        
        if example.test_mode == 0 then
        for j = 1 , example.no_joints do
             if is_flipped == 1 then

                joints[1][j] = joints_warped[1][trainData.flipped[j]]
                joints[2][j] = joints_warped[2][trainData.flipped[j]] 
                is_visible[j] = example.is_visible[1][trainData.flipped[j]] 
             
             else
                
                joints[1][j] = joints_warped[1][j]
                joints[2][j] = joints_warped[2][j] 
                if example.is_train == 1 then
                   is_visible[j] = example.is_visible[1][j]
                end  
             end
        end
       
	end
        if example.test_mode == 0 then
        return  warp, joints, is_flipped, is_visible
        else
        return warp
        end  
end

function get_warpflow(size, warp)


	local grid_y = torch.ger( torch.linspace(1, size, size), torch.ones(size) )
	local grid_x = torch.ger( torch.ones(size), torch.linspace(1, size, size) )
        local flow = torch.FloatTensor()
	flow:resize(3, size, size)
	flow[1] = grid_x
	flow[2] = grid_y
	flow[3] = torch.ones(size, size)
        flow = flow:reshape(3, size * size)
	local warp_inv = torch.inverse(warp)
       
        local warped = torch.mm(warp_inv,flow);
	warped = warped:reshape(3, size, size)
	
        local warpedflow = torch.FloatTensor()
	warpedflow:resize(2, size, size)
	warpedflow[1] = warped[2]
	warpedflow[2] = warped[1]

        return warpedflow

end

function apply_augmentation(example)

        local warp, joints_warped, flipped, is_visible = compute_warp_matrix(example) 
        local size =  example.output_size 
        
        local warpedflow = get_warpflow(size, warp) 
	local example_warped = {}
        local im_warped = image.warp(example.image, warpedflow, 'bicubic', false)
        
        local scale = 0
        if example.test_mode == 0 then
        if opt.dataset == 'mpi' then 
             
             local head_rect_warped =  torch.mm(warp, example.head_rect:float())
             scale = compute_head_size(head_rect_warped)

        else
            
             local torso_warped = torch.mm(warp, example.torso)
             scale = compute_torso_size(torso_warped, flipped) 
             example_warped.torso = torso_warped
        end
        end         
       
        if example.is_train == 1 then
	
           local heatmaps  =  generate_heatmap(joints_warped, example)
           example_warped.gt_maps = heatmaps
           example_warped.is_visible = is_visible
           
        end  

     
       
        if example.returnwarp == 1 then
          example_warped.warp_inv = torch.inverse(warp)
        end
       
     
        example_warped.image = im_warped
        if example.test_mode == 0 then
        example_warped.scale = scale
        example_warped.joints = joints_warped 
        end
        return example_warped
       
  
end

function compute_pckh(heatmaps,joints,scale,th)

   local total_joints = heatmaps:size(1)
   local is_correct = torch.zeros(total_joints)
   
   for j = 1 , total_joints do
   
	local vals,mrows = torch.max(heatmaps[j],2)
   	local val,row = torch.max(vals,1)
   
        local col = mrows[row[1][1]]
        local y_pred = row[1][1]
        local x_pred = col[1]
        
        local dy = (y_pred - joints[2][j])*(y_pred - joints[2][j])
        local dx = (x_pred - joints[1][j])*(x_pred - joints[1][j])
        local d = torch.sqrt((dx + dy))
        d = (d*100)/scale
        if d <= th then
           is_correct[j] = 1
        end
        
  end

   return is_correct

end

function compute_pck(heatmaps, joints, scale, th)

   local total_joints = heatmaps:size(1)
   local is_correct = torch.zeros(total_joints)
   
   for j = 1 , total_joints do
   
	local vals,mrows = torch.max(heatmaps[j],2)
   	local val,row = torch.max(vals,1)
   
        local col = mrows[row[1][1]]
        local y_pred = row[1][1]
        local x_pred = col[1]
       
        local dy = (y_pred - joints[2][j])*(y_pred - joints[2][j])
        local dx = (x_pred - joints[1][j])*(x_pred - joints[1][j])
        local d = torch.sqrt((dx + dy))
        d = (d*100)/scale
        if d <= th then
           is_correct[j] = 1
        end
        
  end

   return is_correct

end


function compute_head_size(rect)


local dx = (rect[1][1] - rect[1][2])*(rect[1][1] - rect[1][2])
local dy = (rect[2][1] - rect[2][2])*(rect[2][1] - rect[2][2])
local d = torch.sqrt((dx + dy))
d = d*0.6;
return d

end

function compute_torso_size(rect,flipped)

local dx = 0
local dy = 0

if flipped == 0 then 

	 dx = (rect[1][1] - rect[1][2])*(rect[1][1] - rect[1][2])
	 dy = (rect[2][1] - rect[2][2])*(rect[2][1] - rect[2][2])
else
	 dx = (rect[1][3] - rect[1][4])*(rect[1][3] - rect[1][4])
	 dy = (rect[2][3] - rect[2][4])*(rect[2][3] - rect[2][4])
end

local d = torch.sqrt((dx + dy))
return d

end

function exponential_decay(params)

local decayed_learning_rate = 0
local tmp = 0
if params.staircase == 1 then

   tmp = torch.floor((params.global_step / params.decay_steps))

else   

   tmp = params.global_step / params.decay_steps

end

decayed_learning_rate = params.learning_rate * torch.pow(params.decay_rate, tmp)

return decayed_learning_rate

end





