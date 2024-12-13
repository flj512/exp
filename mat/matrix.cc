#include<cstring>
#include<cstdio>
#include<cmath>
#include<cassert>
#include<cstdlib>
#include<chrono>
#include"common/utils.h"
#define AVX512F_CHECK defined(__GNUC__)&&defined(__AVX512F__)
#if AVX512F_CHECK
    #include<immintrin.h>
#endif

#define N (1024) // only test NxN matrix

void mat_init(float *m, int n)
{
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            m[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void matmul_base(float* out, const float* A,const float* B, int n, int s)
{
    size_t mat_size = n*n*sizeof(float);
    memset(out,0,mat_size);

    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                out[i*s+j] += A[i*s+k]*B[k*s+j];
}

void matmul_cache_friendly(float* out, const float* A,const float* B, int n, int s)
{
    size_t mat_size = n*n*sizeof(float);
    memset(out,0,mat_size);

    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                out[i*s+k] += A[i*s+j]*B[j*s+k];
}

#if AVX512F_CHECK
void matmul_avx512(float* out, const float* A,const float* B, int n, int s)
{
    /*
        the trunk shape of A matrix is (BLOCK_Y, BLOCK_X)
        the trunk shape of B matrix is (BLOCK_X, AVX_FLOAT_SIZE)
    */
    // 512 bit vector can contain 16 single float number.
    const int AVX_FLOAT_SIZE= 16; 
    // this should be the multiple of cache line size, but should not too large
    // the chunks of B matrix can fit inside the L1 cache.
    const int BLOCK_SIZE_X = 16;  
    // This value can be set larger, but also consider the size of L1.
    const int BLOCK_SIZE_Y = std::min(n/32,64);
    size_t mat_size = n*n*sizeof(float);
    memset(out,0,mat_size);    

    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0;i<n;i+=BLOCK_SIZE_Y){
        for(int j=0;j<n;j+=BLOCK_SIZE_X){
            for(int k=0;k<n;k+=AVX_FLOAT_SIZE){
                for(int m=0;m<BLOCK_SIZE_Y;m++){
                    __m512 o = _mm512_load_ps(&out[(i+m)*s+k]);
                    for(int p=0;p<BLOCK_SIZE_X;p++)
                    {
                        __m512 c = _mm512_set1_ps(A[(i+m)*s+j+p]);
                        __m512 b = _mm512_load_ps(&B[(j+p)*s+k]);
                        o = _mm512_fmadd_ps(c,b,o);
                    }
                    _mm512_store_ps(&out[(i+m)*s+k],o);
                }
            }
        }
    }
}
#endif

void compare_mat(const float* A,const float* B,int n,int s)
{
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            if(fabs(A[i*s+j]-B[i*s+j]) > 1e-5){
                printf("incorrect result!!!\n");
                return;
            }

}

float* make_buffer()
{
    const int buffer_size = N*N*sizeof(float);
    const int aliagn = 64;
    return (float*)std::aligned_alloc(aliagn, buffer_size);    
}

#define BENCHMARK_FUNCTION(C,A,B,func,n,s,BC,CNT) \
{\
    auto start = get_current_time_ms();\
    for(int i=0;i<(CNT);i++)\
        func(C,A,B,n,s);\
    printf(#func" = %0.2f ms\n",(float)(get_current_time_ms()-start)/(CNT));\
    if((BC)!=nullptr) compare_mat(C,BC,n,s);\
}

int main(int argc,char* argv[])
{
    const int N_1024 = N/1024;
    const int DEFAULT_LOOP = 100;
    const int LOOP_CNT = N_1024 <= 1? DEFAULT_LOOP : std::max(1, DEFAULT_LOOP/(N_1024*N_1024*N_1024));

    auto A = make_buffer();
    auto B = make_buffer();
    auto BC= make_buffer();
    auto C= make_buffer();

    mat_init(A,N);
    mat_init(B,N);

    printf("%dx%d matrix multiplication benchmark.\n",N,N);

    if(N_1024<=1){
        BENCHMARK_FUNCTION(BC,A,B,matmul_base,N,N,nullptr,1)
    }

    BENCHMARK_FUNCTION(C,A,B,matmul_cache_friendly,N,N,N_1024<=1?BC:nullptr,LOOP_CNT)

#if AVX512F_CHECK
    BENCHMARK_FUNCTION(BC,A,B,matmul_avx512,N,N,C,LOOP_CNT)
#endif

    std::free(A);
    std::free(B);
    std::free(BC);
    std::free(C);
    return 0;
}