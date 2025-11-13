from collections import deque
import random
import time

big_deque = deque([x for x in range(1_000_000)])

big_list = [x for x in range(1_000_000)]

s = time.perf_counter()
for i in range(10000):
    sample = random.sample(big_deque, 32)

print(f"Big deque t {time.perf_counter() - s}")

s = time.perf_counter()
for i in range(10000):
    sample = random.sample(big_list, 32)
    
print(f"Big list t {time.perf_counter() - s}")