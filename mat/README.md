# Matrix Benchmark
Benchmark 1024x1024 matrix with default threads (Total logic threads/2)  
`./mat/matrix benchmark`

Benchmark 8192x8192 matrix with default threads (Total logic threads/2)  
`./mat/matrix benchmark 8192`

Benchmark 1024x1024 matrix with specific threads  
`OMP_NUM_THREADS=2 ./mat/matrix benchmark`

# Test Result
Test result on my laptop (8 core 16 threads)  
CPU: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz  
RAM: 16GB DDR4 3200MT/S  
OS: Ubuntu 20.04.2 LTS  
Caches (sum of all):      
&nbsp;&nbsp;&nbsp;&nbsp;L1d:                    384 KiB (8 instances)  
&nbsp;&nbsp;&nbsp;&nbsp;L1i:                    256 KiB (8 instances)  
&nbsp;&nbsp;&nbsp;&nbsp;L2:                     10 MiB (8 instances)  
&nbsp;&nbsp;&nbsp;&nbsp;L3:                     24 MiB (1 instance)  

Compiler Clang 14.0.0, OMP 8 threads.  
Pytorch uses 8 threads by default  
## 512x512 matrix benchmark
```
[multiple]
matmul_naive = 19.0590 ms, 1 times
matmul_cache_friendly = 0.8080 ms, 400 times
matmul_openblas = 0.5271 ms, 400 times
matmul_strassen = 0.6081 ms, 400 times
matmul_avx512_block = 0.5918 ms, 400 times
matmul_avx512_block_tiny = 0.5488 ms, 400 times
matmul_avx512_entire = 1.2557 ms, 400 times

[add & sub]
matsub_naive = 0.1752 ms, 1600 times
matadd_naive = 0.1507 ms, 1600 times
matadd_avx512 = 0.1014 ms, 1600 times
matsub_avx512 = 0.1116 ms, 1600 times

[memory]
memcpy speed = 14.6114 GB/s, total Copied 19.5 GB
avxcpy speed = 12.5704 GB/s, total Copied 19.5 GB
```

Pytorch  
```
torch cpu mul 0.686 ms, 400 times
torch cpu add 0.105 ms, 1600 times
torch cpu sub 0.070 ms, 1600 times
```

## 1024x1024 matrix benchmark
```
[multiple]
matmul_naive = 384.6370 ms, 1 times
matmul_cache_friendly = 13.2248 ms, 100 times
matmul_openblas = 2.8794 ms, 100 times
matmul_strassen = 5.0098 ms, 100 times
matmul_avx512_block = 5.7561 ms, 100 times
matmul_avx512_block_tiny = 5.4112 ms, 100 times
matmul_avx512_entire = 13.8323 ms, 100 times

[add & sub]
matsub_naive = 0.6071 ms, 400 times
matadd_naive = 0.5596 ms, 400 times
matadd_avx512 = 0.5721 ms, 400 times
matsub_avx512 = 0.4029 ms, 400 times

[memory]
memcpy speed = 13.6997 GB/s, total Copied 19.5 GB
avxcpy speed = 12.7723 GB/s, total Copied 19.5 GB
```

Pytorch  
```
torch cpu mul 5.666 ms, 100 times
torch cpu add 0.546 ms, 400 times
torch cpu sub 0.308 ms, 400 times
```

## 4096x4096 matrix benchmark
```
[multiple]
matmul_openblas = 281.2790 ms, 1 times
matmul_strassen = 356.8440 ms, 1 times
matmul_avx512_block = 353.1410 ms, 1 times
matmul_avx512_entire = 808.0160 ms, 1 times

[add & sub]
matsub_naive = 10.8257 ms, 25 times
matadd_naive = 9.8787 ms, 25 times
matadd_avx512 = 6.0487 ms, 25 times
matsub_avx512 = 6.1657 ms, 25 times

[memory]
memcpy speed = 15.4672 GB/s, total Copied 19.5 GB
avxcpy speed = 12.4380 GB/s, total Copied 19.5 GB
```

Pytorch  
```
torch cpu mul 273.356 ms, 1 times
torch cpu add 14.261 ms, 25 times
torch cpu sub 17.215 ms, 25 times
```
