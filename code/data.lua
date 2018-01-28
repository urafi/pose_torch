--Script for loading the data
--The test/validation and train set for MPII, LSP and FLIC are in data folder in torch 7 format.
 
	trainData = torch.load('data/train_set_' .. opt.dataset .. '.t7')
	trSize = #trainData

	if opt.dataset == 'mpi' then
        	testData = torch.load('data/validation_set_' .. opt.dataset .. '.t7')
	else  
		testData = torch.load('data/test_set_' .. opt.dataset .. '.t7')
	end
	tsSize = #testData

