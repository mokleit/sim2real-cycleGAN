# GENERATE REALISTIC IMAGES FROM SIMULATED IMAGES USING CYCLEGAN

# Introduction
We provide here a structure to train a PyTorch implementation of CycleGAN. We cloned the original repository and used the provided Colab for our training. Everything specific to CycleGAN can be found [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). What we provide here is a way to reproduce the results of our experiment. 

[**PROJECT FOLDER**](https://drive.google.com/drive/folders/14u7NbWyEez3Pwk2yoFvdjKTWziQwp4F2?usp=sharing). You will most likely need to copy this folder in your Drive in order to be able to run it in Google Colab. 

# Dataset

The dataset is formed of two collection of images, real images and sim images: both zip files can be found in the link provided in the introduction. The images are already preprocessed and split into train and test sets to avoid to facilitate things for the user. They can be found in [**PROJECT FOLDER**](https://drive.google.com/drive/folders/14u7NbWyEez3Pwk2yoFvdjKTWziQwp4F2?usp=sharing) under the following path: **pytorch-CycleGAN-and-pix2pix/datasets/sim2real**. There are 4 folders: trainA, trainB, testA, testB. TrainA and testA correspond. respectively, to the training and test synthetic/simulated images. The same reasoning applies for trainB and testB, which is for real images. If you want to use your own dataset, you will need to upload your two collection of images, one in for each domain (in our case we have simulated vs real images). We provide a Colab notebook in [**PROJECT FOLDER**](https://drive.google.com/drive/folders/14u7NbWyEez3Pwk2yoFvdjKTWziQwp4F2?usp=sharing) under the name **CreateDataset.ipynb**. to guide you in extracting the images and placing them in the correct folder.

# Training

Training is completed by running **CycleGAN_Pytorch** notebook in [**PROJECT FOLDER**](https://drive.google.com/drive/folders/14u7NbWyEez3Pwk2yoFvdjKTWziQwp4F2?usp=sharing). During training, it is possible to see samples of generated images under the following path: **pytorch-CycleGAN-and-pix2pix/checkpoints/sim2real/web/images** (you can take a look at examples of images from previous runs before starting training).

# Testing 

In order to test the model, we use the same notebook as for training (we run last cell). Before doing so, we need to execute a few steps as mentioned in the notebook training part. The test results can be found under the following path: **pytorch-CycleGAN-and-pix2pix/results/sim2real/test_latest/images**


