import numpy as np
np.random.seed(42)  # for reproducibility

import os
os.environ['KERAS_BACKEND'] = 'theano'

from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_flow import batch_generator
from Utils import true_generated_batch
import os.path
import pickle

def make_trainable(model, value):
	for layer in model.layers:
		layer.trainable = value

batch_size = 8000
mini_batch_size = 64
nb_epoch = 2000
train_size = 82611
valid_size = 40438
patience = 50

# input image dimensions
img_rows, img_cols = 64, 64

runMax = 300
epoch=0
run=0
found = False
CPNameAE = None
CPNameD = None

#Look for checkpoint models

for i in range(runMax,-1,-1):
	for j in range(nb_epoch,-1,-1):	
		epochStr = str(j)
		runStr = str(i)

		if (os.path.exists('CP/lsganAEADV/AE/AE-'+runStr+'-'+epochStr+'.h5') and os.path.exists('CP/lsganAEADV/disc/disc-'+runStr+'-'+epochStr+'.h5')):
			CPNameAE = 'CP/lsganAEADV/AE/AE-'+runStr+'-'+epochStr+'.h5'
			CPNameD = 'CP/lsganAEADV/disc/disc-'+runStr+'-'+epochStr+'.h5'
			epoch=j+1
			run=i+1
			found=True
			break
	if found:
		break

if found:

	autoencoder = load_model(CPNameAE)
	discriminator = load_model(CPNameD)
	
	losses = pickle.load(open('CP/lsganAEADV/losses.p', 'rb'))

else:

	# encoder

	encoder = Sequential()

	encoder.add(Conv2D(64, 5, padding='same', strides=(2,2), input_shape=(64, 64, 3))) #64-32
	encoder.add(BatchNormalization())
	encoder.add(Activation('relu'))
	encoder.add(Dropout(0.5))

	encoder.add(Conv2D(128, 5, padding='same', strides=(2,2))) #32-16
	encoder.add(BatchNormalization())
	encoder.add(Activation('relu'))
	encoder.add(Dropout(0.5))

	encoder.add(Conv2D(256, 5, padding='same', strides=(2,2))) #16-8
	encoder.add(BatchNormalization())
	encoder.add(Activation('relu'))
	encoder.add(Dropout(0.5))

	encoder.add(Conv2D(512, 5, padding='same', strides=(2,2))) #8-4
	encoder.add(BatchNormalization())
	encoder.add(Activation('relu'))
	encoder.add(Dropout(0.5))

	encoder.add(Conv2D(512, 5, padding='same', strides=(1,1))) #4-4
	encoder.add(BatchNormalization())
	encoder.add(Activation('relu'))

	encoder.add(Flatten())

	encoder.add(Dense(1024))

	# decoder

	decoder = Sequential()

	decoder.add(Reshape(target_shape=(4, 4, 64), input_shape=(1024,)))

	decoder.add(Conv2DTranspose(512, 5, padding='same', strides=(2,2))) #4-8
	decoder.add(BatchNormalization())
	decoder.add(Activation('relu'))
	decoder.add(Dropout(0.5))

	decoder.add(Conv2DTranspose(256, 5, padding='same', strides=(2,2))) #8-16
	decoder.add(BatchNormalization())
	decoder.add(Activation('relu'))
	decoder.add(Dropout(0.5))

	decoder.add(Conv2DTranspose(128, 5, padding='same', strides=(2,2))) #16-32
	decoder.add(BatchNormalization())
	decoder.add(Activation('relu'))
	decoder.add(Dropout(0.5))

	decoder.add(Conv2DTranspose(3, 5, padding='same', strides=(1,1)))
	decoder.add(Activation('sigmoid'))

	# autoencoder

	iae = layers.Input(shape=(64, 64, 3))
	iencoded = encoder(iae)
	idecoded = decoder(iencoded)
	autoencoder = models.Model(iae, idecoded)

	#merger

	i1 = layers.Input(shape=(32, 32, 3))
	i2 = layers.Input(shape=(64, 64, 3))
	i1p = ZeroPadding2D(padding=(16, 16)) (i1)
	o = layers.add([i1p, i2])
	merger_model = models.Model([i1, i2], o)

	# discriminator

	discriminator = Sequential()

	discriminator.add(merger_model)

	discriminator.add(Conv2D(64, 5, padding='same', strides=(2, 2))) #64-32
	discriminator.add(BatchNormalization())
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.5))

	discriminator.add(Conv2D(64, 5, padding='same', strides=(2, 2))) #32-16
	discriminator.add(BatchNormalization())
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.5))

	discriminator.add(Conv2D(64, 5, padding='same', strides=(2, 2))) #16-8
	discriminator.add(BatchNormalization())
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.5))

	discriminator.add(Conv2D(64, 5, padding='same', strides=(2, 2))) #8-4
	discriminator.add(BatchNormalization())
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.5))

	discriminator.add(Conv2D(512, 5, padding='same'))
	discriminator.add(BatchNormalization())
	discriminator.add(LeakyReLU(alpha=0.2))
	discriminator.add(Dropout(0.5))

	discriminator.add(Flatten())

	discriminator.add(Dense(512))
	discriminator.add(BatchNormalization())
	discriminator.add(LeakyReLU(alpha=0.2))	
	discriminator.add(Dropout(0.5))

	discriminator.add(Dense(1))
	discriminator.add(Activation('sigmoid'))

	losses={}
	losses={'ae_L1':[], 'ae_adv':[], 'disc':[]}

# GAN

igan1 = layers.Input(shape=(64, 64, 3))
igan2 = layers.Input(shape=(64, 64, 3))
generated = autoencoder(igan1)

make_trainable(discriminator, False)

classified = discriminator([generated, igan2])

GAN = models.Model([igan1, igan2], [classified, generated])


#Compiling Models

if found:

	autoencoder.compile(loss='mean_absolute_error', optimizer='adam')

	GAN.compile(loss=['binary_crossentropy', 'mean_absolute_error'], loss_weights=[0.1, 0.9], optimizer='adam')

	make_trainable(discriminator, True)

	discriminator.compile(loss='binary_crossentropy', optimizer='sgd')

else:

	encoder.compile(loss='mean_absolute_error', optimizer='adam')
	decoder.compile(loss='mean_absolute_error', optimizer='adam')
	autoencoder.compile(loss='mean_absolute_error', optimizer='adam')

	GAN.compile(loss=['binary_crossentropy', 'mean_absolute_error'], loss_weights=[0.1, 0.9], optimizer='adam')

	make_trainable(discriminator, True)

	discriminator.compile(loss='binary_crossentropy', optimizer='sgd')


GAN.summary()


number_of_batches = int(np.ceil(train_size/batch_size))

i = epoch

while (i<nb_epoch):

	print('epoch')
	print(i)

	trainDataGenerator = batch_generator(hdf5_name='train_data.hdf',data_size=train_size , batch_size=batch_size)

	for b in range(number_of_batches):

		print('batch')
		print(b)

		context_batch, centers_batch = next(trainDataGenerator)

		actual_batch_size = context_batch.shape[0]
		number_of_mini_batches = int(np.ceil(actual_batch_size/mini_batch_size))

		for j in range(number_of_mini_batches):

			print('minibatch')
			print(j)

			context_mini_batch = context_batch[j*mini_batch_size:min((j+1)*mini_batch_size, actual_batch_size)]
			centers_mini_batch = centers_batch[j*mini_batch_size:min((j+1)*mini_batch_size, actual_batch_size)]

			actual_mini_batch_size = context_mini_batch.shape[0]

			targets_generated = autoencoder.predict(context_mini_batch)

			inputs_discriminator_batch, targets_discriminator_batch = true_generated_batch(centers_mini_batch, targets_generated, labels_smoothing = True)

			make_trainable(discriminator, True)

			d_loss = discriminator.train_on_batch([inputs_discriminator_batch, np.vstack([context_mini_batch, context_mini_batch])], targets_discriminator_batch)

			make_trainable(discriminator, False)

			GAN_targets_batch = np.zeros([actual_mini_batch_size,1])
			GAN_targets_batch[:,0]=1.0

			g_loss = GAN.train_on_batch([context_mini_batch, context_mini_batch], [GAN_targets_batch, centers_mini_batch])

			losses['ae_adv'].append(g_loss[0])
			losses['ae_L1'].append(g_loss[1])
			losses['disc'].append(d_loss)

	pickle.dump(losses, open('CP/lsganAEADV/losses.p', 'wb'))
	autoencoder.save('CP/lsganAEADV/AE/AE-'+str(run)+'-'+str(i)+'.h5')
	discriminator.save('CP/lsganAEADV/disc/disc-'+str(run)+'-'+str(i)+'.h5')

	i+=1

print('Training done')

autoencoder.save('lsganAEADV.h5')
discriminator.save('lsganAEADV_disc.h5')
GAN.save('lsganAEADV_disc.h5')

print('Models Saved')
