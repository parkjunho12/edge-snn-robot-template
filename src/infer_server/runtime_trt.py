"""
TensorRT Runtime for EMG Model Inference
Loads .plan engine file and performs inference on EMG data
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
        
        # Load engine
        self._load_engine()
        self._allocate_buffers()
        
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
        logger.info(f"Number of bindings: {self.engine.num_bindings}")
        
    def _allocate_buffers(self):
        """Allocate GPU memory buffers for inputs and outputs"""
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            
            is_input = self.engine.binding_is_input(i)
            
            logger.info(f"Binding {i}: {binding_name}, shape={shape}, dtype={dtype}, is_input={is_input}")
            
            if is_input:
                self.input_shapes[binding_name] = shape
                self.input_dtypes[binding_name] = dtype
            else:
                self.output_shapes[binding_name] = shape
                self.output_dtypes[binding_name] = dtype
            
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
        for i in range(self.engine.num_bindings):
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
        output_binding_idx = self.engine.num_bindings - 1  # Assuming last binding is output
        output_shape = self.context.get_binding_shape(output_binding_idx)
        output_name = self.engine.get_binding_name(output_binding_idx)
        
        # Reshape output
        output_data = self.outputs[output_binding_idx]['host'].reshape(output_shape)
        
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
        # Free GPU memory
        for inp in self.inputs:
            if inp and inp['device']:
                inp['device'].free()
        
        for out in self.outputs:
            if out and out['device']:
                out['device'].free()
        
        logger.info("TensorRT runtime cleaned up")


def main():
    """Example usage"""
    # Path to engine file
    engine_path = "output/rate/model_tcn_fp16.plan"
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