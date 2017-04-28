# ift6266 - Contextual Image inpainting

# Dependencies
-Keras;

-h5py;

-Numpy;

# Training a model

Convolutional Autoencoder:

  mkdir -p CP/cnnL2AE
  python cnnL2AE.py
  
Convolutional Autoencoder with L1+Adversarial losses:

  mkdir -p CP/autoencADVL1/AE
  mkdir CP/autoencADVL1/disc
  python autoencoderADVL1.py

DCGAN with L1 loss:

  mkdir -p CP/dcgan/gen
  mkdir CP/dcgan/disc
  mkdir CP/dcgan/GAN
  python dcganL1.py

-All the training files require the data set in .hdf as train_data.hdf and valid_data.hdf (two data sets in each file with keys 'inputs' and 'targets')
-dcganL1.py requires a keras pre-trained encoder 'encoder.h5' with output shape = (None, 4, 4, 128)
