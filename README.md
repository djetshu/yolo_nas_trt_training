# YOLO NAS TensorRT Training
This repository facilitates the following tasks:
- Training a YOLO NAS model with custom data (`yolo_train_inference.ipynb`)
- Converting a custom YOLO NAS model to a TensorRT file (`onnx2trt.py`)
- Testing the custom YOLO NAS TensorRT model with a webcam in real-time (`test_trt.py`)

## Demonstration
*Include any demonstration content here, such as images, videos, or links to examples.*

## Installation

### Prerequisites
- **Operating System**: Ubuntu 20.04 or later (Tested on 22.04)
- **NVIDIA CUDA Toolkit**: Version 11.7
- **CuDNN**: Version 8.1.x or later
- **NVIDIA Driver**: Must support CUDA 11.2 or later (Version 460.x or higher)

### Cloning the Repository
To clone the repository, run the following command:
```bash
git clone https://github.com/djetshu/yolo_nas_trt_training.git
```

### Creating the Environment and Installing Requirements
Create an Anaconda environment and install the necessary requirements as specified in the `environment.yml` file.

Key dependencies:
- TensorRT 8.6.1.post1
- PyCUDA 2024.1
- Python 3.9
- PyTorch 1.13.1

To create the environment, run:
```bash
conda env create -f environment.yml
```

## Training
Refer to the notebook `yolo_train_inference.ipynb` for detailed instructions on training the YOLO NAS model with your custom data.

## Converting YOLO NAS to TensorRT
To convert a trained YOLO NAS model to a TensorRT file, use the following command:
```bash
python onnx2trt.py --onnx-file model_export/yolo_nas_s_custom.onnx --trt-output-file model_export/yolo_nas_s_custom.trt
```

## Deploying TensorRT with a Webcam in Real-Time
To test the TensorRT model with a webcam in real-time, run:
```bash
python test_trt.py --model-path model_export/yolo_nas_s_custom.trt
```

## Workflow During and After Training

### Training on PC
It is highly recommended to train on a PC with decent GPU capabilities (e.g., GTX 1060 or better). Due to the large size of the required packages in this Anaconda environment, it is not recommended to train on the final deployment hardware, such as Jetson AGX Orin, due to memory restrictions.

### Conversion of ONNX to TensorRT
This step is performed to increase the processing speed of the model. Perform this conversion on the PC that will execute the final model (e.g., Jetson AGX Orin, PC, etc.).

### Running on the Final PC or SBC
The final model can be run as a simple Python script or as a ROS2 node, depending on your deployment requirements.

## Contact Information

For inquiries, collaboration opportunities, or questions feel free to contact:

- **Email:** daffer.queque@outlook.com
