import statistics
import time
from typing import Callable, Dict


def bench(fn: Callable[[], None], warmup: int = 5, iters: int = 50) -> Dict[str, float]:
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
