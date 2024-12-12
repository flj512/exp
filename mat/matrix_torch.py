import torch
import time

N = 1024
a = torch.randn(N,N)
b = torch.randn(N,N)

LOOP_CNT = 50

start = time.perf_counter()
for i in range(LOOP_CNT):
    c = a.matmul(b)
print(f"torch cpu {(time.perf_counter()-start)*1000/LOOP_CNT:0.2f} ms")
