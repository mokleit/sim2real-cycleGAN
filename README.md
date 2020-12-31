#REPRODUCE CYCLE GAN RESULTS

Everything was performed on google colab. In order to avoid the headache of collecting the dataset and preparing it to be used by cycleGan, we provide a google drive folder containing everything needed in order to reproduce the results.

Some files in that folder would only be useful if you wanted to create your own dataset and use our model on this. 

The link to the shared folder: https://drive.google.com/drive/folders/14u7NbWyEez3Pwk2yoFvdjKTWziQwp4F2?usp=sharing

In order to reproduce the results, simply run the following notebook: CycleGAN_Pytorch.

NOTE: You'll have to copy the folder in your own drive and change the path in that file to reflect the its location in your drive.

Once the training is completed. The test results can be found at the following path:
pytorch-CycleGAN-and-pix2pix/results/sim2real/test_latest/images

