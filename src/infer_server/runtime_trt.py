"""
TensorRT Runtime for EMG Model Inference
Loads .plan engine file and performs inference on EMG data

Supports TensorRT 8.x, 9.x, and 10.x with automatic API version detection
"""

import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect TensorRT version
TRT_VERSION = trt.__version__
TRT_MAJOR = int(TRT_VERSION.split('.')[0])
logger.info(f"TensorRT version: {TRT_VERSION} (major: {TRT_MAJOR})")

# Version-specific compatibility flags
USE_V2_API = TRT_MAJOR >= 10  # TensorRT 10+ uses new binding API
SUPPORT_TENSORS = TRT_MAJOR >= 10  # TensorRT 10+ supports tensor I/O

def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT Runtime for EMG Model Inference"
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default=None,
        help="Path to Plan file (default: <artifact-dir>/model_<prefix>_fp16.plan)",
    )

    return parser.parse_args()

class TRTRuntime:
    """TensorRT runtime for model inference"""
    
    def __init__(self, engine_path: str, logger_severity: trt.Logger.Severity = trt.Logger.WARNING):
        """
        Initialize TensorRT runtime
        
        Args:
            engine_path: Path to the .plan engine file
            logger_severity: TensorRT logger severity level
        """
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        # Initialize TensorRT
        self.logger = trt.Logger(logger_severity)
        self.runtime = None
        self.engine = None
        self.context = None
        
        # TensorRT version info
        self.trt_version = TRT_VERSION
        self.trt_major = TRT_MAJOR
        self.use_v2_api = USE_V2_API
        
        # Memory buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        
        # Input/output metadata
        self.input_shapes = {}
        self.output_shapes = {}
        self.input_dtypes = {}
        self.output_dtypes = {}
        self.input_names = []
        self.output_names = []
        
        # TensorRT 10+ specific
        self.tensor_names = []
        self.input_tensors = {}
        self.output_tensors = {}
        
        # Load engine
        self._load_engine()
        self._allocate_buffers()
        
        logger.info(f"TensorRT runtime initialized (v{self.trt_version}, API v{'2' if self.use_v2_api else '1'})")
        
    def _load_engine(self):
        """Load serialized engine from file"""
        logger.info(f"Loading TensorRT engine from {self.engine_path}")
        
        # Create runtime
        self.runtime = trt.Runtime(self.logger)
        
        # Read engine file
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Deserialize engine
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        logger.info(f"Engine loaded successfully")
        num_bindings = getattr(self.engine, "num_bindings", None)
        if num_bindings is None:
            num_bindings = self.engine.num_io_tensors
            
        self.num_bindings = num_bindings
        
        logger.info(f"Number of bindings: {self.num_bindings}")
        
    def _allocate_buffers(self):
        """Allocate GPU memory buffers for inputs and outputs"""
        self.stream = cuda.Stream()
        
        if self.use_v2_api:
            # TensorRT 10+ uses new tensor I/O API
            logger.info("Using TensorRT 10+ tensor I/O API")
            self._allocate_buffers_v10()
        else:
            # TensorRT 8.x/9.x uses binding API
            logger.info("Using TensorRT 8.x/9.x binding API")
            self._allocate_buffers_v8()
    
    def _allocate_buffers_v8(self):
        """Allocate buffers for TensorRT 8.x/9.x (legacy binding API)"""
        for i in range(self.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            
            is_input = self.engine.binding_is_input(i)
            
            logger.info(f"Binding {i}: {binding_name}, shape={shape}, dtype={dtype}, is_input={is_input}")
            
            if is_input:
                self.input_shapes[binding_name] = shape
                self.input_dtypes[binding_name] = dtype
                self.input_names.append(binding_name)
            else:
                self.output_shapes[binding_name] = shape
                self.output_dtypes[binding_name] = dtype
                self.output_names.append(binding_name)
            
            # For dynamic shapes, we'll allocate later based on actual input
            if -1 in shape:
                logger.info(f"Dynamic shape detected for {binding_name}, will allocate on first inference")
                self.bindings.append(None)
                if is_input:
                    self.inputs.append(None)
                else:
                    self.outputs.append(None)
            else:
                # Allocate memory
                size = trt.volume(shape)
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if is_input:
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def _allocate_buffers_v10(self):
        """Allocate buffers for TensorRT 10+ (new tensor I/O API)"""
        # Get all tensor names
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            self.tensor_names.append(tensor_name)
            
            # Get tensor properties
            try:
                dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                shape = self.engine.get_tensor_shape(tensor_name)
                mode = self.engine.get_tensor_mode(tensor_name)
                
                is_input = (mode == trt.TensorIOMode.INPUT)
                
                logger.info(f"Tensor {i}: {tensor_name}, shape={shape}, dtype={dtype}, mode={mode}")
                
                if is_input:
                    self.input_shapes[tensor_name] = shape
                    self.input_dtypes[tensor_name] = dtype
                    self.input_names.append(tensor_name)
                else:
                    self.output_shapes[tensor_name] = shape
                    self.output_dtypes[tensor_name] = dtype
                    self.output_names.append(tensor_name)
                
                # For dynamic shapes, allocate later
                if -1 in shape:
                    logger.info(f"Dynamic shape detected for {tensor_name}, will allocate on first inference")
                    if is_input:
                        self.input_tensors[tensor_name] = None
                    else:
                        self.output_tensors[tensor_name] = None
                else:
                    # Allocate memory
                    size = np.prod(shape)
                    host_mem = cuda.pagelocked_empty(int(size), dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    if is_input:
                        self.input_tensors[tensor_name] = {'host': host_mem, 'device': device_mem}
                    else:
                        self.output_tensors[tensor_name] = {'host': host_mem, 'device': device_mem}
                        
            except Exception as e:
                logger.warning(f"Error processing tensor {tensor_name}: {e}")
    
    def _allocate_dynamic_buffers(self, input_data: np.ndarray, input_name: str = 'emg'):
        """Allocate buffers for dynamic shapes based on actual input"""
        input_shape = input_data.shape
        
        # Set dynamic input shape
        binding_idx = self.engine.get_binding_index(input_name)
        self.context.set_binding_shape(binding_idx, input_shape)
        
        # Allocate input buffer if needed
        if self.inputs[binding_idx] is None:
            size = np.prod(input_shape)
            dtype = self.input_dtypes[input_name]
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.inputs[binding_idx] = {'host': host_mem, 'device': device_mem}
            self.bindings[binding_idx] = int(device_mem)
        
        # Allocate output buffers based on inferred shapes
        for i in range(self.num_bindings):
            if not self.engine.binding_is_input(i) and self.outputs[i] is None:
                output_shape = self.context.get_binding_shape(i)
                output_name = self.engine.get_binding_name(i)
                
                if -1 not in output_shape:
                    size = np.prod(output_shape)
                    dtype = self.output_dtypes[output_name]
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    self.outputs[i] = {'host': host_mem, 'device': device_mem}
                    self.bindings[i] = int(device_mem)
                    
                    logger.info(f"Allocated output buffer for {output_name}: shape={output_shape}")
    
    def infer(self, input_data: np.ndarray, input_name: str = 'emg') -> np.ndarray:
        """
        Run inference on input data
        
        Args:
            input_data: Input numpy array with shape matching model input (e.g., [batch, sequence, channels])
            input_name: Name of the input binding (default: 'emg')
        
        Returns:
            Output numpy array
        """
        if self.use_v2_api:
            return self._infer_v10(input_data, input_name)
        else:
            return self._infer_v8(input_data, input_name)
    
    def _infer_v8(self, input_data: np.ndarray, input_name: str = 'emg') -> np.ndarray:
        """Inference for TensorRT 8.x/9.x using binding API"""
        # Ensure input is the correct dtype
        input_binding_idx = self.engine.get_binding_index(input_name)
        expected_dtype = self.input_dtypes[input_name]
        
        if input_data.dtype != expected_dtype:
            input_data = input_data.astype(expected_dtype)
        
        # Handle dynamic shapes
        if self.inputs[input_binding_idx] is None:
            self._allocate_dynamic_buffers(input_data, input_name)
        
        # Flatten input data
        input_data_flat = input_data.ravel()
        
        # Copy input to host buffer
        np.copyto(self.inputs[input_binding_idx]['host'], input_data_flat)
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(
            self.inputs[input_binding_idx]['device'],
            self.inputs[input_binding_idx]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Transfer predictions back from GPU
        for i, output in enumerate(self.outputs):
            if output is not None:
                cuda.memcpy_dtoh_async(
                    output['host'],
                    output['device'],
                    self.stream
                )
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Get output shape
        output_binding_idx = self.num_bindings - 1  # Assuming last binding is output
        output_shape = self.context.get_binding_shape(output_binding_idx)
        output_name = self.engine.get_binding_name(output_binding_idx)
        
        # Reshape output
        output_data = self.outputs[output_binding_idx]['host'].reshape(output_shape)
        
        return output_data.copy()
    
    def _infer_v10(self, input_data: np.ndarray, input_name: str = 'emg') -> np.ndarray:
        """Inference for TensorRT 10+ using tensor I/O API"""
        # Ensure input is the correct dtype
        expected_dtype = self.input_dtypes[input_name]
        
        if input_data.dtype != expected_dtype:
            input_data = input_data.astype(expected_dtype)
        
        input_shape = tuple(input_data.shape)
        
        # Set input shape for dynamic dimensions
        if -1 in self.input_shapes[input_name]:
            self.context.set_input_shape(input_name, input_shape)
        
        # Allocate input buffer if needed
        if input_name not in self.input_tensors or self.input_tensors[input_name] is None:
            size = np.prod(input_shape)
            host_mem = cuda.pagelocked_empty(int(size), expected_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.input_tensors[input_name] = {'host': host_mem, 'device': device_mem}
        
        # Copy input data to host buffer
        input_data_flat = input_data.ravel()
        np.copyto(self.input_tensors[input_name]['host'], input_data_flat)
        
        # Transfer input data to GPU
        cuda.memcpy_htod_async(
            self.input_tensors[input_name]['device'],
            self.input_tensors[input_name]['host'],
            self.stream
        )
        
        # Set tensor address
        self.context.set_tensor_address(input_name, int(self.input_tensors[input_name]['device']))
        
        # Allocate output buffers based on inferred shapes
        for output_name in self.output_names:
            output_shape = self.context.get_tensor_shape(output_name)
            
            if output_name not in self.output_tensors or self.output_tensors[output_name] is None:
                if -1 not in output_shape:
                    size = np.prod(output_shape)
                    dtype = self.output_dtypes[output_name]
                    host_mem = cuda.pagelocked_empty(int(size), dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.output_tensors[output_name] = {'host': host_mem, 'device': device_mem}
                    logger.info(f"Allocated output buffer for {output_name}: shape={output_shape}")
            
            # Set output tensor address
            if output_name in self.output_tensors and self.output_tensors[output_name] is not None:
                self.context.set_tensor_address(output_name, int(self.output_tensors[output_name]['device']))
        
        # Execute inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Transfer output data from GPU
        output_name = self.output_names[0]  # Get first output
        output_shape = self.context.get_tensor_shape(output_name)
        
        cuda.memcpy_dtoh_async(
            self.output_tensors[output_name]['host'],
            self.output_tensors[output_name]['device'],
            self.stream
        )
        
        # Synchronize stream
        self.stream.synchronize()
        
        # Reshape and return output
        output_data = self.output_tensors[output_name]['host'].reshape(output_shape)
        
        return output_data.copy()
    
    def infer_batch(self, input_batch: np.ndarray, input_name: str = 'emg') -> np.ndarray:
        """
        Run inference on a batch of inputs
        
        Args:
            input_batch: Batch of input arrays
            input_name: Name of the input binding
        
        Returns:
            Batch of output arrays
        """
        return self.infer(input_batch, input_name)
    
    def warmup(self, input_shape: Tuple[int, ...], num_iterations: int = 10, input_name: str = 'emg'):
        """
        Warm up the engine with dummy data
        
        Args:
            input_shape: Shape of input data
            num_iterations: Number of warmup iterations
            input_name: Name of the input binding
        """
        logger.info(f"Warming up engine with {num_iterations} iterations...")
        
        dtype = self.input_dtypes[input_name]
        dummy_input = np.random.randn(*input_shape).astype(dtype)
        
        for i in range(num_iterations):
            _ = self.infer(dummy_input, input_name)
        
        logger.info("Warmup complete")
    
    def get_input_shape(self, input_name: str = 'emg') -> Tuple[int, ...]:
        """Get expected input shape"""
        return self.input_shapes.get(input_name)
    
    def get_output_shape(self, output_name: Optional[str] = None) -> Tuple[int, ...]:
        """Get output shape"""
        if output_name is None:
            # Return first output shape
            output_name = list(self.output_shapes.keys())[0]
        return self.output_shapes.get(output_name)
    
    def benchmark(self, input_shape: Tuple[int, ...], num_iterations: int = 100, 
                  warmup_iterations: int = 10, input_name: str = 'emg') -> dict:
        """
        Benchmark inference performance
        
        Args:
            input_shape: Shape of input data
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations
            input_name: Name of the input binding
        
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Warmup
        self.warmup(input_shape, warmup_iterations, input_name)
        
        # Prepare input
        dtype = self.input_dtypes[input_name]
        dummy_input = np.random.randn(*input_shape).astype(dtype)
        
        # Benchmark
        logger.info(f"Running benchmark with {num_iterations} iterations...")
        latencies = []
        
        for i in range(num_iterations):
            start_time = time.perf_counter()
            _ = self.infer(dummy_input, input_name)
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        latencies = np.array(latencies)
        
        results = {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'std_latency_ms': np.std(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p90_latency_ms': np.percentile(latencies, 90),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_qps': 1000.0 / np.mean(latencies)
        }
        
        logger.info("Benchmark Results:")
        logger.info(f"  Mean Latency: {results['mean_latency_ms']:.3f} ms")
        logger.info(f"  Median Latency: {results['median_latency_ms']:.3f} ms")
        logger.info(f"  P99 Latency: {results['p99_latency_ms']:.3f} ms")
        logger.info(f"  Throughput: {results['throughput_qps']:.2f} QPS")
        
        return results
    
    def __del__(self):
        """Clean up resources"""
        try:
            # Free GPU memory for legacy API
            for inp in self.inputs:
                if inp and 'device' in inp and inp['device']:
                    inp['device'].free()
            
            for out in self.outputs:
                if out and 'device' in out and out['device']:
                    out['device'].free()
            
            # Free GPU memory for tensor API (TensorRT 10+)
            if hasattr(self, 'input_tensors'):
                for tensor in self.input_tensors.values():
                    if tensor and 'device' in tensor and tensor['device']:
                        tensor['device'].free()
            
            if hasattr(self, 'output_tensors'):
                for tensor in self.output_tensors.values():
                    if tensor and 'device' in tensor and tensor['device']:
                        tensor['device'].free()
            
            logger.info("TensorRT runtime cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")


def main():
    """Example usage"""
    # Path to engine file
    args = parse_args()
    # Initialize runtime
    runtime = TRTRuntime(args.engine_path)
    
    
    # Print input/output info
    print(f"Input shapes: {runtime.input_shapes}")
    print(f"Output shapes: {runtime.output_shapes}")
    
    # Example inference with shape [1, 200, 16] (batch=1, sequence=200, channels=16)
    input_shape = (1, 200, 16)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Run inference
    output = runtime.infer(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0, :5]}")  # Print first 5 values
    
    # Benchmark
    results = runtime.benchmark(input_shape, num_iterations=100)
    print("\nBenchmark Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()