# yolo_nas_trt_training
Environment to train and test fast Yolo NAS to be exported as onnx ready to convert to TensorRT


# Prerequisites
## General requirements to train locally

    Python 3.7, 3.8 or 3.9 installed.
    1.9.0 <= torch < 1.14 
    Nvidia CUDA Toolkit >= 11.2
    CuDNN >= 8.1.x
    Nvidia Driver with CUDA >= 11.2 support (≥460.x)

## Current requirements

    Nvidia CUDA Toolkit = 11.7
    CuDNN >= 8.1.x
    Nvidia Driver with CUDA >= 11.2 support (≥460.x)
    +
    Requirements inside of ancaconda environment
    tensorrt==8.6.1.post1
    pycuda==2024.1
    Python 3.9
    torch = 1.13.1
