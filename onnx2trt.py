from typing import List, Tuple
import tensorrt as trt
import argparse
import pycuda.driver as cuda
import pycuda.autoinit

print(trt.__version__)
device = cuda.Device(0)
device.compute_capability()

def convert_onnx_to_trt_engine(onnx_file, trt_output_file, enable_int8_quantization:bool = False):
  trt_logger = trt.Logger(trt.Logger.VERBOSE)
  EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

  with trt.Builder(trt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, builder.create_builder_config() as config:

    config = builder.create_builder_config()
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    if enable_int8_quantization:
      config.set_flag(trt.BuilderFlag.INT8)
    else:
      config.set_flag(trt.BuilderFlag.FP16)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Load your ONNX model
    with trt.OnnxParser(network, trt_logger) as onnx_parser:
      with open(onnx_file, 'rb') as f:
        parse_success = onnx_parser.parse(f.read())
        if not parse_success:
          errors = "\n".join(
              [str(onnx_parser.get_error(error)) for error in range(onnx_parser.num_errors)]
          )
          raise RuntimeError(f"Failed to parse onnx model for trt conversion. Errors: \n{errors}")

      trt_logger.log(trt.ILogger.INFO, "Parsed ONNX model")

    # Query input names and shapes from parsed TensorRT network
    network_inputs = [network.get_input(i) for i in range(network.num_inputs)]
    input_names = [_input.name for _input in network_inputs]  # ex: ["actual_input1"]

    assert input_names[0] == 'input' # Sometimes this line of code should be commented to convert to TRT

    serialized_engine = builder.build_serialized_network(network, config)
    with open(trt_output_file, "wb") as output_file:
      output_file.write(serialized_engine)
      trt_logger.log(trt.ILogger.INFO, "Serialization done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ONNX model to TensorRT engine')
    parser.add_argument('--onnx-file', type=str, required=True, help='Path to the ONNX model file')
    parser.add_argument('--trt-output-file', type=str, required=True, help='Path to save the TensorRT engine file')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 quantization')

    args = parser.parse_args()
    convert_onnx_to_trt_engine(args.onnx_file, args.trt_output_file, enable_int8_quantization=args.int8)