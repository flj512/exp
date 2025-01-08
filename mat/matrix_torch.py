import torch
import time

# Test result on my laptop (8 core 16 threads), pytorch uses 8 threads by default
# CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz
# RAM: 16GB DDR4 3200MHz
# OS: Ubuntu 20.04.2 LTS
# Caches (sum of all):      
#   L1d:                    384 KiB (8 instances)
#   L1i:                    256 KiB (8 instances)
#   L2:                     10 MiB (8 instances)
#   L3:                     24 MiB (1 instance)

# 128x128 matrix benchmark
# torch cpu mul 0.015 ms, 6400 times
# torch cpu add 0.013 ms, 25600 times
# torch cpu sub 0.010 ms, 25600 times

# 256x256 matrix benchmark
# torch cpu mul 0.093 ms, 1600 times
# torch cpu add 0.023 ms, 6400 times
# torch cpu sub 0.024 ms, 6400 times

# 512x512 matrix benchmark
# torch cpu mul 0.686 ms, 400 times
# torch cpu add 0.105 ms, 1600 times
# torch cpu sub 0.070 ms, 1600 times

# 1024x1024 matrix benchmark
# torch cpu mul 5.666 ms, 100 times
# torch cpu add 0.546 ms, 400 times
# torch cpu sub 0.308 ms, 400 times

# 2048x2048 matrix benchmark
# torch cpu mul 40.594 ms, 12 times
# torch cpu add 2.326 ms, 100 times
# torch cpu sub 2.187 ms, 100 times

# 4096x4096 matrix benchmark
# torch cpu mul 273.356 ms, 1 times
# torch cpu add 14.261 ms, 25 times
# torch cpu sub 17.215 ms, 25 times

N = 1024
N_1024 = N/1024

def mat_benchmark(op,cnt:int, tag:str)->None:
    A = [torch.randn(N,N) for _ in range(cnt)]
    B = [torch.randn(N,N) for _ in range(cnt)]
    C = [torch.randn(N,N) for _ in range(cnt)]
    start = time.perf_counter()
    for a,b,c in zip(A,B,C):
        c = op(a,b)
    print(f"torch cpu {tag} {(time.perf_counter()-start)*1000/cnt:0.3f} ms, {cnt} times")

print(f"{N}x{N} matrix benchmark")

MUL_CNT = 100
loop_cnt = MUL_CNT/(N_1024*N_1024) if N_1024 <= 1 else max(1,MUL_CNT/(N_1024*N_1024*N_1024))
mat_benchmark(lambda a,b: a.matmul(b),int(loop_cnt),"mul")

ADD_CNT = 400
loop_cnt = int(max(1,ADD_CNT//(N_1024*N_1024)))
mat_benchmark(lambda a,b: a+b,loop_cnt,"add")
mat_benchmark(lambda a,b: a-b,loop_cnt,"sub")
