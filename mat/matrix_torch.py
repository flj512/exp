import torch
import time

N = 1024
a = torch.randn(N,N)
b = torch.randn(N,N)

N_1024 = N//1024
DEFAULT_CNT = 100
LOOP_CNT = 100 if N_1024 <= 1 else max(1,100//(N_1024*N_1024*N_1024))

a.matmul(b)

start = time.perf_counter()
for i in range(LOOP_CNT):
    c = a.matmul(b)

print(f"torch cpu mul {(time.perf_counter()-start)*1000/LOOP_CNT:0.3f} ms")

start = time.perf_counter()
for i in range(DEFAULT_CNT):
    c = a + b

print(f"torch cpu add {(time.perf_counter()-start)*1000/DEFAULT_CNT:0.3f} ms")
