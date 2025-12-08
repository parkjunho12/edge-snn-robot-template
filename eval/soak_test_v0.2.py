# eval/soak_test_v0_2s.py
import time
import requests
from statistics import mean

API_URL = "http://localhost:8000/infer"  # 실제 엔드포인트로 교체
REQUESTS_PER_SECOND = 2
DURATION_MINUTES = 30
TIMEOUT_SECONDS = 2.0  # 요청 타임아웃


def main():
    total_requests = 0
    success = 0
    errors = 0
    timeouts = 0
    latencies = []

    start = time.time()
    end = start + DURATION_MINUTES * 60

    print(f"[SoakTest] Start: {time.ctime(start)}")
    print(f"[SoakTest] Target: {REQUESTS_PER_SECOND} req/s for {DURATION_MINUTES} min")

    while time.time() < end:
        loop_start = time.time()
        for _ in range(REQUESTS_PER_SECOND):
            total_requests += 1
            t0 = time.time()
            try:
                # 여기에 실제 EMG dummy payload 넣기
                payload = {
                    "encoding_type": "rate",
                    "model_prefix": "tcn",
                    "device": "cpu",
                }
                r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
                latency = (time.time() - t0) * 1000.0  # ms

                if r.status_code == 200:
                    success += 1
                    latencies.append(latency)
                else:
                    errors += 1
                    print(f"[WARN] HTTP {r.status_code}: {r.text[:200]}")
            except requests.exceptions.Timeout:
                timeouts += 1
                print("[WARN] Request timeout")
            except Exception as e:
                errors += 1
                print(f"[ERROR] {e}")

        # 1초 주기 맞추기
        elapsed = time.time() - loop_start
        sleep_time = max(0.0, 1.0 - elapsed)
        time.sleep(sleep_time)

    duration = time.time() - start
    dropout = (errors + timeouts) / max(1, total_requests) * 100.0

    print("\n================ Soak Test Result (v0.2s) ================")
    print(f"Duration (s):          {duration:.2f}")
    print(f"Total requests:        {total_requests}")
    print(f"Success:               {success}")
    print(f"Errors:                {errors}")
    print(f"Timeouts:              {timeouts}")
    print(f"Dropout rate:          {dropout:.3f}% (target < 0.5%)")

    if latencies:
        print(f"Mean latency (ms):     {mean(latencies):.3f}")
        print(f"Min latency (ms):      {min(latencies):.3f}")
        print(f"Max latency (ms):      {max(latencies):.3f}")
    print("==========================================================")


if __name__ == "__main__":
    main()
