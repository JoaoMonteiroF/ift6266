import h5py
import numpy as np

'''
Given the resized data (64x64), this method builts a 
minibatch of images with square black holes in the center.
A list of corrupted images is returned in inputs, respective 
targets in a list named targets. 
Also, the list captions returns a set of captions for each
image.

For train = 1, images are from the training data;
for train = 0, images are from the validation data.

all = True if batch_size = data set length
In this case, batch_idx = 0

'''


def batch_generator(hdf5_name, data_size, batch_size=32):
	
	number_of_slices = int(np.ceil(data_size/batch_size))
	
	while True:
		open_file = h5py.File(hdf5_name, 'r')

		for i in xrange(0, number_of_slices):
			inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
			targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]
       			
       			yield (inputs_batch, targets_batch)

		open_file.close()			


def batch_generator_with_embeddings(hdf5_name, hdf5_name_captions_emb, data_size, load_captions_emb=True, batch_size=32):
	
	number_of_slices = int(np.ceil(data_size/batch_size))
	
	if (load_captions_emb):
		while True:
			open_file_cap_emb = h5py.File(hdf5_name_captions_emb, 'r')
			open_file = h5py.File(hdf5_name, 'r')

			for i in xrange(0, number_of_slices):
				caps_emb_batch = open_file_cap_emb['emb'][i*batch_size:min((i+1)*batch_size, data_size)]
				caps_emb_batch_out = np.zeros([batch_size, 300])

				for j in xrange(0, batch_size):
					rand_num = np.random.randint(0, 5)
					caps_emb_batch_out[j, :] = caps_emb_batch[i, rand_num, :]
        			
				inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
				targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]	
        			
        			yield (inputs_batch, targets_batch, caps_emb_batch_out)

			open_file.close()
			open_file_cap_emb.close()


	else:
		while True:
			open_file = h5py.File(hdf5_name, 'r')

			for i in xrange(0, number_of_slices):
				inputs_batch = open_file['inputs'][i*batch_size:min((i+1)*batch_size, data_size)]
				targets_batch = open_file['targets'][i*batch_size:min((i+1)*batch_size, data_size)]
        			
        			yield (inputs_batch, targets_batch)

			open_file.close


def batch_generator_GAN(train_or_valid, hdf5_name, batch_size=32):
	
	if train_or_valid == 1:
		number_of_batches = int(round(82611/batch_size))
	else:
		number_of_batches = int(round(40438/batch_size))	
	
	while True:
		open_file = h5py.File(hdf5_name, 'r')

		for i in xrange(0, number_of_batches):
			inputs_batch = open_file['complete'][i*batch_size:(i+1)*batch_size]

        		yield (inputs_batch)	

		open_file.close()

if __name__ == '__main__':
	gen = batch_generator_with_embeddings(hdf5_name='train_data.hdf', hdf5_name_captions_emb = 'embeddings_train_norm.hdf', data_size=82611)		
	a, b, c = next(gen)
	print(a.shape)
	print(b.shape)
	print(c.shape)

