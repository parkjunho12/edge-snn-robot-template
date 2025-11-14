import time
import statistics
from typing import Callable


def bench(fn: Callable, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": statistics.mean(ts),
        "p95_ms": statistics.quantiles(ts, n=100)[94],
    }
