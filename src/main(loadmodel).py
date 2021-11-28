models_path = '../res/models/'
weights_path = '../res/weights/'
dataset_path = '../res/dataset/'

import time
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import best_loss, magic_accuracy

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#model = tf.keras.models.load_model(models_path + 'papier_icip_qp_22.h5', custom_objects={'best_loss': best_loss, 'magic_accuracy' : magic_accuracy})
model = tf.keras.models.load_model(models_path + 'model_tech_db_filtered2020-05-13.h5')

#model.summary()

image = np.array([np.load('../res/dataset/1000008.npy')], dtype=np.float32)

qp = np.array([22], dtype=np.float32)

elapsed_times = []

for i in tqdm(range(1000)):

	start_time = time.time()

	#prediction = model.predict([image])
	prediction = model.predict([image,qp])

	end_time = time.time()

	elapsed_times.append((end_time - start_time) * 1000)

print('Inferences finished ! Dimensions : ', prediction.shape, ' / Average execution time : %.3f ms ' % (sum(elapsed_times)/len(elapsed_times)), sep="")




