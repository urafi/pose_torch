#  pose_torch
This repository contains the code for the Human Pose Estimation framework described in the paper [An Efficient Convolutional Network for Human Pose Estimation](http://pages.iai.uni-bonn.de/gall_juergen/download/jgall_posecnn_bmvc16).

## Getting Started

This code requires Torch 7 and cudnn to run. Please make sure both of them are installed on the system.

Download the MPI dataset and place the images folder in  folder ../Images/MPI/ so that the path to images is ../Images/MPI/images/

Do the same for FLIC, LSP and extended LSP datasets.

After that the training procedure is fairly straighforward.
On the command line execute :

th do_all.lua -dataset 'mpi'

to start training on the mpi dataset. 

Replace 'mpi' with 'flic' or 'lsp' for training on lsp or flic datasets.

The do_all.lua scripts runs the following scripts required for training and validation.

1) data.lua to load train and test/validation data for a dataset.
2) model.lua to load the model.
3) loss.lua to load the BCE loss.
4) train.lua contains the train() function that runs one epoch over the entire train set.
5) test.lua  contains the test() function that runs one epoch over the entire test/validation set.
 
The do_all.lua script than calls train() and test() functions total_epoch times(total number of epochs to run).

As the training progresses the code saves the best scoring model  and the accuracy per epoch in the progress folder.

We also provide a script run_model.lua to evaluate the pre-trained models. Download the compressed pre-trained models folder and decompress it. Then execute :

th run_model.lua -dataset 'lsp'

To evaluate the pre-trained model for lsp. For mpi and flic simply replace 'lsp' with 'mpi' and 'flic' respectively. The script also generates a matfile containing 2D joints predictions for the testset for a dataset and saves it in the predictions folder.

## Notes

1. In the paper, the training time reported was with an older version of cuDNN, and after switching to cuDNN 4, training time may improve.
2. For convenience during training , the code shows PCK and PCKh for all joints. However there might be discrepency between numbers when officially evaluated on FLIC, MPII or LSP, but  still it provides a good picture of how well the network is learning during training.

## Questions?

Any questions or comments are welomed at rafi@vision.rwth-aachen.de
