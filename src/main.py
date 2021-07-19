weights_path = '../res/weights/'
dataset_path = '../res/dataset/'

import time
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from CNN import model_technicolor_vector_multi_qp

import numpy as np
from tqdm import tqdm

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

with tf.device('/GPU:0'):

	model = model_technicolor_vector_multi_qp()

	# Either load the weights from the h5 file, or load them from the native format using the expect_partial() method of load_weights(...)

	model.load_weights(weights_path + 'weights.h5')
	#model.load_weights(weights_path + 'my_weights').expect_partial()

	#model.summary()

	image = np.array([np.load('../res/dataset/1000008.npy')])

	qp = np.array([22])

	elapsed_times = []

	for i in tqdm(range(10000)):

		start_time = time.time()

		prediction = model.predict([image,qp])

		end_time = time.time()

		elapsed_times.append((end_time - start_time) * 1000)

	print('Inferences finished ! Dimensions : ', prediction.shape, ' / Average execution time : %.3f ms ' % (sum(elapsed_times)/len(elapsed_times)), sep="")
