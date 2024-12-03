import torch
import time


a = torch.randn(1024,1024)
b = torch.randn(1024,1024)

for i in range(10):
    a.matmul(b)

start = time.perf_counter()
c = a.matmul(b)
print(f"torch cpu {int((time.perf_counter()-start)*1000)} ms")
