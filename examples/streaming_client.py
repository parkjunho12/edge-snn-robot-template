"""
Example client for testing the streaming inference API
"""

import json
import time
import requests
from typing import Optional
import argparse


class StreamingInferenceClient:
    """Client for consuming streaming inference results"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self) -> dict:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stream_stats(self) -> dict:
        """Get streaming statistics"""
        response = requests.get(f"{self.base_url}/infer/stream/stats")
        response.raise_for_status()
        return response.json()
    
    def single_inference_pytorch(self, batch: int = 1, channels: int = 16, 
                                 length: int = 200, steps: int = 1) -> dict:
        """Run single inference with PyTorch"""
        response = requests.post(
            f"{self.base_url}/infer",
            json={
                "batch": batch,
                "channels": channels,
                "length": length,
                "steps": steps
            }
        )
        response.raise_for_status()
        return response.json()
    
    def single_inference_tensorrt(self, batch: int = 1, channels: int = 16, 
                                  length: int = 200) -> dict:
        """Run single inference with TensorRT"""
        response = requests.post(
            f"{self.base_url}/infer/tensorrt",
            json={
                "batch": batch,
                "channels": channels,
                "length": length,
                "steps": 1
            }
        )
        response.raise_for_status()
        return response.json()
    
    def stream_inference(
        self,
        duration_seconds: Optional[float] = None,
        use_tensorrt: bool = False,
        fps: int = 30,
        preprocess: bool = True,
        callback = None
    ):
        """
        Stream inference results
        
        Args:
            duration_seconds: Duration to stream (None = infinite)
            use_tensorrt: Use TensorRT backend if available
            fps: Target frames per second
            preprocess: Apply preprocessing
            callback: Function to call for each result
        """
        config = {
            "duration_seconds": duration_seconds,
            "use_tensorrt": use_tensorrt,
            "fps": fps
        }
        
        response = requests.post(
            f"{self.base_url}/infer/stream",
            json=config,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # Remove 'data: ' prefix
                    try:
                        data = json.loads(data_str)
                        
                        if callback:
                            callback(data)
                        else:
                            # Default: print to console
                            if 'error' in data:
                                print(f"ERROR: {data['error']}")
                            elif 'status' in data and data['status'] == 'completed':
                                print(f"\n{'='*60}")
                                print(f"Stream completed:")
                                print(f"  Total frames: {data['total_frames']}")
                                print(f"  Duration: {data['duration_seconds']:.2f}s")
                                print(f"  Average FPS: {data['avg_fps']:.2f}")
                                print(f"{'='*60}")
                            else:
                                print(f"Frame {data['frame']:4d} | "
                                      f"Pred: {data['prediction']} | "
                                      f"Conf: {data['confidence']:.2%} | "
                                      f"Latency: {data['latency_ms']:6.2f}ms | "
                                      f"Backend: {data['backend']:10s} | "
                                      f"CPU: {data['cpu_percent']:5.1f}%")
                    
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON: {e}")
        
    def benchmark(self, num_iterations: int = 1000) -> dict:
        """Run benchmark"""
        response = requests.post(
            f"{self.base_url}/benchmark",
            params={"num_iterations": num_iterations}
        )
        response.raise_for_status()
        return response.json()


def example_health_check():
    """Example: Health check"""
    print("\n" + "="*60)
    print("Example 1: Health Check")
    print("="*60)
    
    client = StreamingInferenceClient()
    health = client.health_check()
    print(f"\nServer status: {health['status']}")
    print(f"TensorRT available: {health['tensorrt_available']}")
    print(f"EMG mode: {health['emg_mode']}")


def example_stream_stats():
    """Example: Get stream statistics"""
    print("\n" + "="*60)
    print("Example 2: Stream Statistics")
    print("="*60)
    
    client = StreamingInferenceClient()
    stats = client.get_stream_stats()
    
    print(f"\nStream Configuration:")
    print(f"  TensorRT available: {stats['tensorrt_available']}")
    print(f"  EMG mode: {stats['emg_mode']}")
    print(f"  Window size: {stats['emg_window_size']}")
    print(f"  Channels: {stats['emg_channels']}")
    print(f"  Sampling rate: {stats['emg_sampling_rate']} Hz")
    print(f"  Max throughput: {stats['max_throughput_qps']} QPS")
    print(f"  Expected latency: {stats['expected_latency_ms']} ms")


def example_single_inference():
    """Example: Single inference"""
    print("\n" + "="*60)
    print("Example 3: Single Inference")
    print("="*60)
    
    client = StreamingInferenceClient()
    
    # PyTorch inference
    print("\nPyTorch inference:")
    result_pt = client.single_inference_pytorch()
    print(f"  Latency: {result_pt['latency_ms']} ms")
    print(f"  Spikes: {result_pt['spikes']}")
    print(f"  Shape: {result_pt['shape']}")
    
    # TensorRT inference
    try:
        print("\nTensorRT inference:")
        result_trt = client.single_inference_tensorrt()
        print(f"  Latency: {result_trt['latency_ms']} ms")
        print(f"  Prediction: {result_trt['prediction']}")
        print(f"  Confidence: {result_trt['confidence']}")
        print(f"  Backend: {result_trt['backend']}")
    except requests.exceptions.HTTPError as e:
        print(f"  TensorRT not available: {e}")


def example_streaming_inference():
    """Example: Streaming inference"""
    print("\n" + "="*60)
    print("Example 4: Streaming Inference (10 seconds)")
    print("="*60)
    
    client = StreamingInferenceClient()
    
    # Stream for 10 seconds
    client.stream_inference(
        duration_seconds=10.0,
        use_tensorrt=False,
        fps=30,
        preprocess=True
    )


def example_streaming_with_callback():
    """Example: Streaming with custom callback"""
    print("\n" + "="*60)
    print("Example 5: Streaming with Custom Callback")
    print("="*60)
    
    # Custom callback to collect statistics
    frame_count = 0
    latencies = []
    predictions = []
    
    def custom_callback(data):
        nonlocal frame_count, latencies, predictions
        
        if 'error' in data:
            print(f"ERROR: {data['error']}")
        elif 'status' in data and data['status'] == 'completed':
            print(f"\nStatistics:")
            print(f"  Total frames: {len(latencies)}")
            print(f"  Mean latency: {sum(latencies)/len(latencies):.2f}ms")
            print(f"  Max latency: {max(latencies):.2f}ms")
            print(f"  Min latency: {min(latencies):.2f}ms")
            
            from collections import Counter
            pred_counts = Counter(predictions)
            print(f"  Prediction distribution: {dict(pred_counts)}")
        else:
            frame_count += 1
            latencies.append(data['latency_ms'])
            predictions.append(data['prediction'])
            
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames, "
                      f"avg latency: {sum(latencies[-10:])/10:.2f}ms")
    
    client = StreamingInferenceClient()
    client.stream_inference(
        duration_seconds=5.0,
        use_tensorrt=True,
        fps=30,
        callback=custom_callback
    )


def example_benchmark():
    """Example: Benchmark"""
    print("\n" + "="*60)
    print("Example 6: Benchmark (1000 iterations)")
    print("="*60)
    
    client = StreamingInferenceClient()
    
    print("\nRunning benchmark...")
    results = client.benchmark(num_iterations=1000)
    
    print("\nPyTorch Results:")
    if results['pytorch']:
        pt = results['pytorch']
        print(f"  Mean latency: {pt['mean_latency_ms']:.3f} ms")
        print(f"  Median latency: {pt['median_latency_ms']:.3f} ms")
        print(f"  P99 latency: {pt['p99_latency_ms']:.3f} ms")
        print(f"  Throughput: {pt['throughput_qps']:.2f} QPS")
    
    print("\nTensorRT Results:")
    if results['tensorrt']:
        trt = results['tensorrt']
        print(f"  Mean latency: {trt['mean_latency_ms']:.3f} ms")
        print(f"  Median latency: {trt['median_latency_ms']:.3f} ms")
        print(f"  P99 latency: {trt['p99_latency_ms']:.3f} ms")
        print(f"  Throughput: {trt['throughput_qps']:.2f} QPS")
        
        if results['pytorch']:
            speedup = results['pytorch']['mean_latency_ms'] / trt['mean_latency_ms']
            print(f"\nSpeedup: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Streaming Inference Client")
    parser.add_argument('--url', default='http://localhost:8000', 
                       help='Server URL')
    parser.add_argument('--example', type=int, choices=range(1, 7),
                       help='Run specific example (1-6)')
    parser.add_argument('--stream', action='store_true',
                       help='Run streaming inference')
    parser.add_argument('--duration', type=float, default=10.0,
                       help='Stream duration in seconds')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target frames per second')
    parser.add_argument('--no-tensorrt', action='store_true',
                       help='Disable TensorRT backend')
    
    args = parser.parse_args()
    
    # Update client URL
    StreamingInferenceClient.base_url = args.url
    
    if args.stream:
        # Quick streaming mode
        print(f"Streaming inference for {args.duration}s at {args.fps} FPS...")
        client = StreamingInferenceClient(args.url)
        client.stream_inference(
            duration_seconds=args.duration,
            use_tensorrt=False,
            fps=args.fps
        )
    elif args.example:
        # Run specific example
        examples = {
            1: example_health_check,
            2: example_stream_stats,
            3: example_single_inference,
            4: example_streaming_inference,
            5: example_streaming_with_callback,
            6: example_benchmark,
        }
        examples[args.example]()
    else:
        # Run all examples
        print("\n" + "="*60)
        print("Streaming Inference Client Examples")
        print("="*60)
        
        try:
            example_health_check()
            example_stream_stats()
            example_single_inference()
            example_streaming_inference()
            example_streaming_with_callback()
            example_benchmark()
            
            print("\n" + "="*60)
            print("All examples completed!")
            print("="*60)
        except Exception as e:
            print(f"\nError: {e}")
            print("\nMake sure the server is running:")
            print("  uvicorn src.infer_server.app:app --reload")


if __name__ == "__main__":
    main()