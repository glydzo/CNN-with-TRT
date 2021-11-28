models_path = '../res/models/'
dataset_path = '../res/dataset/'

from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = models_path + 'SavedModel/'
output_saved_model_dir = models_path + 'OptimizedModel/'

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
conversion_params = conversion_params._replace(precision_mode="FP32")
conversion_params = conversion_params._replace(maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir) #,conversion_params=conversion_params)
graph_def = converter.convert()
converter.save(output_saved_model_dir)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow as tf

graph_def = tf.GraphDef()

def my_input_fn():
	image = np.array([np.load(dataset_path + '1000008.npy')]).astype(np.float32)
	print(image.shape)
	qp = np.array([22]).astype(np.float32)
	Inp1 = np.random.normal(size=(1, 68, 68, 1)).astype(np.float32)
  	
	yield ([Inp1,qp])

converter.build(input_fn=my_input_fn)

saved_model_loaded = tf.saved_model.load(output_saved_model_dir)
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
output = frozen_func(input_data)[0].numpy()

