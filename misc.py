
import os
os.environ['KERAS_BACKEND'] = 'theano'

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import plot_model
import h5py
from PIL import Image
import pickle
np.random.seed(42)


def visualize(img_input, img_target, img_pred, save = True, save_file = 'test.jpg'):
	for idx in range(0, img_pred.shape[0]):

		center = (int(np.floor(img_input[idx].shape[0] / 2.)), int(np.floor(img_input[idx].shape[1] / 2.)))
	
		# True full image
		true_full = np.copy(img_input[idx])
		true_full = true_full
		true_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target[idx]
	
		# Predicted full image
		pred_full = np.copy(img_input[idx])
		pred_full = pred_full
		pred_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred[idx] 

		print(img_pred[idx])

		plt.subplot(2, img_pred.shape[0], idx+1)
		plt.imshow(true_full)
		#plt.title('True')
		plt.axis('off')

		plt.subplot(2, img_pred.shape[0], idx+img_pred.shape[0]+1)
		plt.imshow(pred_full)
		#plt.title('Pred')
		plt.axis('off')

	if save:	
		plt.savefig(save_file, bbox_inches='tight')	
	plt.show()

def just_save(title_keyword, img_input, img_target, img_pred, save_file):
	
	title_keyword = 'Epoch' + str(title_keyword)

	for idx in range(0, img_pred.shape[0]):

		center = (int(np.floor(img_input[idx].shape[0] / 2.)), int(np.floor(img_input[idx].shape[1] / 2.)))
	
		# True full image
		true_full = np.copy(img_input[idx])
		true_full = true_full
		true_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_target[idx]
	
		# Predicted full image
		pred_full = np.copy(img_input[idx])
		pred_full = pred_full
		pred_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = img_pred[idx] 

		plt.subplot(2, img_pred.shape[0], idx+1)
		plt.imshow(true_full)
		#plt.title('True')
		plt.axis('off')

		plt.subplot(2, img_pred.shape[0], idx+img_pred.shape[0]+1)
		plt.imshow(pred_full)
		#plt.title('Pred')
		plt.axis('off')

	#plt.suptitle(title_keyword)
	plt.savefig(save_file, bbox_inches='tight')	
	#plt.show()	

def plot_keras_model(modelFileName, shapes=True):

	model = load_model(modelFileName)
	plot_model(model, to_file='model.png', show_shapes = shapes) 	 

def load_model_visualize(model, save = True, save_file = 'gan.jpg', number_of_images = 5, valid_data = 'valid_data.hdf', gan = False):
	
	# Load the model
	model = load_model(model)

	start = 723

	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	inputs_samples = open_file['inputs'][start:start+number_of_images]
	targets_samples = open_file['targets'][start:start+number_of_images]
	open_file.close() 

	# Predict outputs
	if gan:
		encoder = load_model('encoder.h5')
		noise = np.random.normal(0., 0.2, (number_of_images, 4, 4, 128))
		inputs = encoder.predict(inputs_samples)
		inputs += noise
		predict_samples = model.predict(inputs)


	else:
		predict_samples = model.predict(inputs_samples)


	visualize(inputs_samples, targets_samples, predict_samples, save, save_file)

def load_model_visualize_with_emb(model, save = False, save_file = 'test.jpg', number_of_images = 5, valid_data = 'train_data.hdf', captions_data = 'embeddings_train_norm.hdf', gan = False):
	
	# Load the model
	model = load_model(model)

	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	open_file_emb = h5py.File(captions_data, 'r')
	inputs_samples = open_file['inputs'][1005:1005+number_of_images]
	targets_samples = open_file['targets'][1005:1005+number_of_images]
	captions_samples = open_file_emb['emb'][1005:1005+number_of_images]

	captions_samples_selected = np.zeros([number_of_images, 300])

	for i in xrange(0, number_of_images):
		rand_num = np.random.randint(0, 5)
		captions_samples_selected[i, :] = captions_samples[i, rand_num, :]

	open_file.close() 
	open_file_emb.close()
	# Predict outputs
	
	if gan:
		encoder = load_model('encoder.h5')
		noise = np.random.normal(0., 0.2, (number_of_images, 4, 4, 128))
		inputs = encoder.predict(inputs_samples)
		inputs += noise
		predict_samples = model.predict([inputs, captions_samples_selected])
	else:
		predict_samples = model.predict(inputs_samples)


	visualize(inputs_samples, targets_samples, predict_samples, save, save_file)

def load_model_save_results_per_epoch(base_model, number_of_epochs, save_file = 'dcganL1.jpg', number_of_images = 5, valid_data = 'valid_data.hdf', gan=True):
	
	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	inputs_samples = open_file['inputs'][100:100+number_of_images]
	targets_samples = open_file['targets'][100:100+number_of_images]
	open_file.close()

	# Load the model
	for epc in xrange(0, number_of_epochs): 
		if epc < 10:
			model = base_model + '-0-'+ str(epc) + '.h5'
		else:
			model = base_model + '-0-'+ str(epc) + '.h5'
		
		model = load_model(model)	

		if gan:
			encoder = load_model('encoder.h5')
			noise = np.random.normal(0., 0.2, (number_of_images, 4, 4, 128))
			inputs = encoder.predict(inputs_samples)
			inputs += noise
			predict_samples = model.predict(inputs)

		else:
			predict_samples = model.predict(inputs_samples)

		save_file_epc = save_file + '_epoch_' + str(epc) + '.jpg'

		just_save(epc, inputs_samples, targets_samples, predict_samples, save_file_epc)	

def save_image_many_epochs(base_model, epochs, image = 10000, save_file = 'test.jpg', valid_data = 'valid_data.hdf'):
	
	# Read samples from the validation data
	open_file = h5py.File(valid_data, 'r')
	x = open_file['inputs'][image:image+1]
	open_file.close()


	count = 1
	plt.subplot(1, len(epochs)+1, count)
	plt.imshow(x[0])
	plt.axis('off')

	for i in epochs: 
		if i < 10:
			model = base_model + '-0-0'+ str(i) + '.h5'
		else:
			model = base_model + '-0-'+ str(i) + '.h5'
		
		model = load_model(model)
		#x.reshape(1, 64, 64, 3)
		generated = model.predict(x)


		center = (int(np.floor(x[0].shape[0] / 2.)), int(np.floor(x[0].shape[1] / 2.)))
		# Predicted full image
		x_gen = np.copy(x[0])
		x_gen[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = generated 

		count += 1
		plt.subplot(1, len(epochs)+1, count)
		plt.imshow(x_gen)
		plt.axis('off')

	plt.savefig(save_file, bbox_inches='tight')

def load_two_models_save_results(model1, model2, save_file = '2models.jpg', number_of_images = 4, valid_data = 'valid_data.hdf'):
	
	model1 = load_model(model1)
	model2 = load_model(model2)

	open_file = h5py.File(valid_data, 'r')
	inputs_samples = open_file['inputs'][20000:20000+number_of_images]
	targets_samples = open_file['targets'][20000:20000+number_of_images]
	open_file.close()

	generated_1 = model1.predict(inputs_samples)
	generated_2 = model2.predict(inputs_samples)

	for idx in range(0, number_of_images):


		center = (int(np.floor(inputs_samples[idx].shape[0] / 2.)), int(np.floor(inputs_samples[idx].shape[1] / 2.)))
	
		# True full image
		true = np.copy(inputs_samples[idx])
		true[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = targets_samples[idx]
		
		# Predicted full image 1
		full1 = np.copy(inputs_samples[idx])
		full1[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = generated_1[idx]
	
		# Predicted full image 2
		full2 = np.copy(inputs_samples[idx])
		full2[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = generated_2[idx]


		plt.subplot(3, number_of_images, idx+1)
		plt.imshow(true)
		plt.title('True')
		plt.axis('off')

		plt.subplot(3, number_of_images, idx+number_of_images+1)
		plt.imshow(full1)
		plt.title('1st model')
		plt.axis('off')

		plt.subplot(3, number_of_images, idx+2*number_of_images+1)
		plt.imshow(full2)
		plt.title('2nd model')
		plt.axis('off')

	plt.savefig(save_file, bbox_inches='tight')

def save_3_images(model1, model2, img_index = 10000, data = 'valid_data.hdf'):

	open_file = h5py.File(data, 'r')
	context = open_file['inputs'][10000]
	target = open_file['targets'][10000]
	open_file.close()

	#model1 = load_model(model1)
	model2 = load_model(model2)
	

	center = (int(np.floor(context.shape[0] / 2.)), int(np.floor(context.shape[1] / 2.)))
	
	# True full image
	true_full = np.copy(context)
	true_full = true_full
	true_full[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = target
	
	# Predicted AE
	#pred_AE = model1.predict(context)
	#full_AE = np.copy(context)
	#full_AE[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = pred_AE 

	# Predicted GAN
	encoder = load_model('encoder.h5')
	noise = np.random.normal(0., 0.2, (1, 4, 4, 128))
	inputs = encoder.predict(context.reshape(1, 64, 64, 3))
	inputs += noise
	pred_GAN = model2.predict(inputs)
	full_GAN = np.copy(context)
	full_GAN[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = pred_GAN 

	#plt.imshow(true_full)
	#plt.axis('off')
	im = Image.fromarray(np.uint8(255*true_full))
	im.save('True.png')
	#plt.savefig('True', bbox_inches='tight')		

	#plt.imshow(full_AE)
	#plt.axis('off')
	#plt.savefig('AEL1', bbox_inches='tight')		

	#plt.imshow(full_GAN)
	#plt.axis('off')
	#plt.savefig('lsganL1', bbox_inches='tight')		

	im = Image.fromarray(np.uint8(255*full_GAN))
	im.save('lsganL1.png')

def plot_loss(pkl = 'losses.p'):
	losses = pickle.load(file(pkl))
	for key in losses:
		to_plot = losses[key]
		plt.plot(to_plot)
		print(key)
		
	plt.legend(losses.keys())	
	plt.show()	


def true_generated_batch(img_center_true, img_center_generated, labels_smoothing = False):
	batch_size = img_center_true.shape[0]
	inputs_discriminator = np.vstack([img_center_true, img_center_generated])
	targets_discriminator = np.zeros([2*batch_size, 1])

	if (labels_smoothing):
		targets_discriminator[0:batch_size, 0] = np.random.uniform(0.7, 0.99, (batch_size))	# true image: target = (1,0)
		targets_discriminator[batch_size:, 0] = np.random.uniform(0.001, 0.01, (batch_size)) # generated image: target = (0,1)
	else:
		targets_discriminator[0:batch_size, 0] = 1.0	# true image: target = (1,0)
		
	return inputs_discriminator, targets_discriminator

if __name__ == '__main__':
	# Results 1
	#load_model_visualize('AE-2-58.h5', gan=False)
	#load_model_visualize_with_emb('gen-0-0-half.h5', gan=True)
	#load_model_save_results_per_epoch('/home/isabela/Desktop/lsgan/gen/gen', 16)
	#plot_loss(pkl ='losses.p')
	#save_3_images('cnnL1AE.h5', '/home/isabela/Desktop/lsgan/gen/gen-0-15'.h5')
	plot_keras_model('decoder_plot.h5')
