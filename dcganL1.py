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

def make_trainable(model, value):
	for layer in model.layers:
		layer.trainable = value

batch_size = 8000
mini_batch_size = 32 
nb_epoch = 2000
train_size = 82611
valid_size = 40438

runMax = 300
epoch=0
run=0
found = False
CPNameG = None
CPNameD = None

#Look for checkpoint models

for i in range(runMax,-1,-1):
	for j in range(nb_epoch,-1,-1):	
		epochStr = str(j)
		runStr = str(i)

		if (os.path.exists('CP/lsgan/gen/gen-'+runStr+'-'+epochStr+'.h5') and os.path.exists('CP/lsgan/disc/disc-'+runStr+'-'+epochStr+'.h5')):
			CPNameG = 'CP/lsgan/gen/gen-'+runStr+'-'+epochStr+'.h5'
			CPNameD = 'CP/lsgan/disc/disc-'+runStr+'-'+epochStr+'.h5'
			epoch=j+1
			run=i+1
			found=True
			break
	if found:
		break

if found:

	generator = load_model(CPNameG)
	discriminator = load_model(CPNameD)

	losses = pickle.load(open('losses.p', 'rb'))

	# Importing image context encoder

	encoder = load_model('encoder.h5')
	make_trainable(encoder, False)

	igan1 = layers.Input(shape=(64, 64, 3))
	igan2 = layers.Input(shape=(64, 64, 3))
	encoded = encoder(igan1)
	generated = generator(encoded)

	make_trainable(discriminator, False)

	classified = discriminator([generated, igan2])

	GAN = models.Model([igan1, igan2], classified)

else:

	# generator

	generator = Sequential()
	
	generator.add(Conv2DTranspose(512, 5, padding='same', strides=(2,2), input_shape=(4, 4, 128))) #4-8
	generator.add(BatchNormalization())
	generator.add(Activation('relu'))
	generator.add(Dropout(0.5))

	generator.add(Conv2DTranspose(256, 5, padding='same', strides=(2,2))) #8-16
	generator.add(BatchNormalization())
	generator.add(Activation('relu'))
	generator.add(Dropout(0.5))

	generator.add(Conv2DTranspose(128, 5, padding='same', strides=(2,2))) #16-32
	generator.add(BatchNormalization())
	generator.add(Activation('relu'))
	generator.add(Dropout(0.5))

	generator.add(Conv2DTranspose(3, 5, padding='same', strides=(1,1)))
	generator.add(Activation('sigmoid'))

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


	# Importing image context encoder

	encoder = load_model('encoder.h5')
	make_trainable(encoder, False)

	# GAN

	igan1 = layers.Input(shape=(64, 64, 3))
	igan2 = layers.Input(shape=(64, 64, 3))
	encoded = encoder(igan1)
	generated = generator(encoded)

	make_trainable(discriminator, False)

	classified = discriminator([generated, igan2])

	GAN = models.Model([igan1, igan2], classified)

	losses = {}
	losses={'gen':[], 'l1_loss':[], 'disc':[]}


#Compiling Models

if found:

	GAN.compile(loss='binary_crossentropy', optimizer='adam')

else:

	GAN.compile(loss='binary_crossentropy', optimizer='adam')

	generator.compile(loss='mean_absolute_error', optimizer='adam')

	make_trainable(discriminator, True)

	discriminator.compile(loss='binary_crossentropy', optimizer='sgd')


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

			noise = np.random.normal(0., 0.2, (actual_mini_batch_size, 4, 4, 128))

			encoded_context = encoder.predict(context_mini_batch)

			input_generator = encoded_context + noise

			targets_generated = generator.predict(input_generator)

			inputs_discriminator_batch, targets_discriminator_batch = true_generated_batch(centers_mini_batch, targets_generated, labels_smoothing = True)

			make_trainable(discriminator, True)

			d_loss = discriminator.train_on_batch([inputs_discriminator_batch, np.vstack([context_mini_batch, context_mini_batch])], targets_discriminator_batch)

			make_trainable(discriminator, False)

			l1_loss = generator.train_on_batch(encoded_context, centers_mini_batch)

			GAN_targets_batch = np.zeros([actual_mini_batch_size,1])
			GAN_targets_batch[:,0]=1.0

			g_loss = GAN.train_on_batch([context_mini_batch, context_mini_batch], GAN_targets_batch)

			losses['gen'].append(g_loss)
			losses['disc'].append(d_loss)
			losses['l1_loss'].append(l1_loss)

		if (b == np.floor(number_of_batches/2)):
			generator.save('CP/lsgan/gen/gen-'+str(run)+'-'+str(i)+'-'+'half'+'.h5')
			pickle.dump(losses, open('CP/lsgan/losses.p', 'wb'))
	
	pickle.dump(losses, open('CP/lsgan/losses.p', 'wb'))
	generator.save('CP/lsgan/gen/gen-'+str(run)+'-'+str(i)+'.h5')
	discriminator.save('CP/lsgan/disc/disc-'+str(run)+'-'+str(i)+'.h5')
	GAN.save('CP/lsgan/GAN/GAN-'+str(run)+'-'+str(i)+'.h5')

	i+=1

print('Training done')

generator.save('lsgan_gen.h5')
discriminator.save('lsgan_disc.h5')
GAN.save('lsgan_GAN.h5')

print('Models Saved')
