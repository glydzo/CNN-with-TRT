# Requirements

Cuda 11.0 update 1 : [link](https://developer.nvidia.com/cuda-11.0-update1-download-archive)

-> Use the local runfile install (you have to previously disable Nouveau driver and unistall all Nvidia drivers).

cuDNN 8.2.1 : [link](https://developer.nvidia.com/rdp/cudnn-archive)

-> Choose Download cuDNN v8.2.1 (June 7th, 2021), for CUDA 11.x

TensorRT 8.0 GA : [link](https://developer.nvidia.com/nvidia-tensorrt-8x-download)

-> Download TensorRT 8.0.1 GA for Linux x86_64 and CUDA 11.3 TAR package and follow the [setup guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar).

# Python setting up

First, be sure you have Python 3.6 installed (if you are using ubuntu 18.04, it is native).

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
python -m pip install uff-0.6.9-py2.py3-none-any.whl
python -m pip install graphsurgeon-0.4.5-py2.py3-none-any.whl
python -m pip install onnx_graphsurgeon-0.2.6-py2.py3-none-any.whl
python -m pip install tensorflow==2.4.0
python -m pip install tqdm
```

# Useful links

[Leveraging TensorFlow-TensorRT integration for Low latency Inference](https://blog.tensorflow.org/2021/01/leveraging-tensorflow-tensorrt-integration.html)

[TensorRT 5 Execution Sample](https://github.com/ktkrhr/tensorrt-sample)

https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#network_python

https://gitlab.insa-rennes.fr/Alexandre.Tissier2/network_partition

https://www.tensorflow.org/install/gpu

https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#prereqs
