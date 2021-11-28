models_path = '../res/models/'
weights_path = '../res/weights/'
dataset_path = '../res/dataset/'

nb_ctus_fullhd = 510
nb_ctus_4k = 2176
nb_max_images_to_process = 10

import time
import os

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
from cnn import model_technicolor_vector_multi_qp

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from bcolors import bcolors

print(bcolors.BOLD + '\nInitialization...\n' + bcolors.ENDC)

print(bcolors.OKGREEN + 'Num GPUs Available: ' + str(len(tf.config.experimental.list_physical_devices('GPU'))) + bcolors.ENDC)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

model = model_technicolor_vector_multi_qp()
#model = tf.saved_model.load(models_path + 'OptimizedModel/')
#infer = model.signatures["serving_default"]

print(bcolors.OKGREEN + 'Model created!' + bcolors.ENDC)

# Either load the weights from the h5 file, or load them from the native format using the expect_partial() method of load_weights(...)

#model.load_weights(weights_path + 'weights.h5')
#model.load_weights(weights_path + 'my_weights').expect_partial()

print(bcolors.OKGREEN + 'Weights loaded!' + bcolors.ENDC)

#model = tf.keras.models.load_model(models_path + 'model_tech_db_filtered2020-05-13.h5')

model.summary()
#tf.keras.models.save_model(model, models_path + 'SavedModel/')

print(bcolors.BOLD + '\nData preparation...\n' + bcolors.ENDC)

image = []
qp = []

print(bcolors.WARNING + 'Shape of input CTUs : ' + str(np.load(dataset_path + '1000008.npy').shape) + bcolors.ENDC)

for i in tqdm(range(nb_ctus_fullhd * nb_max_images_to_process)):

	#image.append(np.load(dataset_path + '1000008.npy'))
	image.append(np.random.rand(68,68,1))
	qp.append(22)

image = np.array(image)
qp = np.array(qp)

quantity_to_input = np.linspace(1, (nb_ctus_fullhd * nb_max_images_to_process), (nb_ctus_fullhd * nb_max_images_to_process) * 100)

print(bcolors.OKGREEN + 'Loading of ' + str(nb_ctus_fullhd * nb_max_images_to_process) + ' CTUs (' + str(nb_max_images_to_process) + ' images FullHD) finished!' + bcolors.ENDC)

elapsed_times = []

with tf.device('/GPU:0'):

	print(bcolors.BOLD + '\nStart of inferences...\n' + bcolors.ENDC)

	for i in range(len(quantity_to_input)):

		print(bcolors.OKBLUE + 'Prediction ' + str(i+1) + ' : ' + str(round(quantity_to_input[i])) + ' CTUs at one time.' + bcolors.ENDC)

		start_time = time.time()

		prediction = model.predict([image[:round(quantity_to_input[i])],qp[:round(quantity_to_input[i])]])
		#x = [image[:round(quantity_to_input[i])],qp[:round(quantity_to_input[i])]]
		#labeling = infer(x)

		end_time = time.time()

		elapsed_times.append((end_time - start_time)*1000/round(quantity_to_input[i]))

		#print(bcolors.OKGREEN,'Prediction finished ! Dimensions : ',prediction.shape,' / Execution time : %.3f ms'%((end_time - start_time)*1000/round(quantity_to_input[i])),sep="")
		print(bcolors.OKGREEN,'Prediction finished ! / Execution time : %.3f ms'%((end_time - start_time)*1000/round(quantity_to_input[i])),sep="")
		print(bcolors.ENDC)

print(bcolors.WARNING,'\nAll predictions are finished !\nAverage throughput : %.3f CTU/s or %.3f FPS (full HD)\n'%((1000/elapsed_times[len(elapsed_times)-1]),((1000/nb_ctus_fullhd)/elapsed_times[len(elapsed_times)-1])),sep="")

plt.plot(quantity_to_input, elapsed_times)
plt.title('Inference time as a function of the number of CTUs passed in input')
plt.xlabel('Nb of CTUs as input')
plt.ylabel('Execution time (in ms)')

plt.show()



