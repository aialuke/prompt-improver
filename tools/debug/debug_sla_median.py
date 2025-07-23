#!/usr/bin/env python3

import statistics

# Same data as the test
response_times = (
    [25] * 35 + [40] * 35 +  # 70% fast (median will be 40ms)
    [75] * 20 + [95] * 6 +   # 26% medium (P95 will be around 95ms)
    [130] * 3 +              # 3% slow
    [250]                    # 1% very slow
)

print(f"Total values: {len(response_times)}")
print(f"Data distribution:")
print(f"  25ms x 35 (positions 0-34)")
print(f"  40ms x 35 (positions 35-69)") 
print(f"  75ms x 20 (positions 70-89)")
print(f"  95ms x 6 (positions 90-95)")
print(f"  130ms x 3 (positions 96-98)")
print(f"  250ms x 1 (position 99)")

sorted_times = sorted(response_times)
print(f"\nValue at position 49 (50th value): {sorted_times[49]}ms")
print(f"Value at position 50 (51st value): {sorted_times[50]}ms")
print(f"Manual median: {(sorted_times[49] + sorted_times[50]) / 2}ms")
print(f"statistics.median(): {statistics.median(response_times)}ms")

median = statistics.median(response_times)
ratio = median / 50.0  # Target is 50ms

print(f"\nTarget: 50ms")
print(f"Actual median: {median}ms")
print(f"Ratio: {ratio:.2f}")
print(f"Status: {'MEETING' if ratio < 0.9 else 'AT_RISK' if ratio < 1.0 else 'BREACHING'}")

# Test P95 and P99 calculations
sorted_times = sorted(response_times)
p95_index = int(len(sorted_times) * 0.95)  # 100 * 0.95 = 95
p99_index = int(len(sorted_times) * 0.99)  # 100 * 0.99 = 99

print(f"\nP95 calculation:")
print(f"  Index: {p95_index} (value: {sorted_times[p95_index]}ms)")
print(f"  Target: 100ms")
print(f"  Status: {'MEETING' if sorted_times[p95_index] <= 100 else 'BREACHING'}")

print(f"\nP99 calculation:")  
print(f"  Index: {p99_index} (value: {sorted_times[p99_index]}ms)")
print(f"  Target: 200ms")
print(f"  Status: {'MEETING' if sorted_times[p99_index] <= 200 else 'BREACHING'}")