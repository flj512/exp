import torch
import time

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
