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

#define N (1024)

void mat_init(float m[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            m[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void matmul_base(float out[N][N], const float A[N][N],const float B[N][N])
{
    size_t mat_size = N*N*sizeof(float);
    memset(out,0,mat_size);

    #pragma omp parallel for
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++)
                out[i][j] += A[i][k]*B[k][j];
}

void matmul_cache_friendly(float out[N][N], const float A[N][N],const float B[N][N])
{
    size_t mat_size = N*N*sizeof(float);
    memset(out,0,mat_size);

    #pragma omp parallel for
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++)
                out[i][k] += A[i][j]*B[j][k];
}

#if AVX512F_CHECK
void matmul_avx512(float out[N][N], const float A[N][N],const float B[N][N])
{
    // 512 bit vector can contain 16 single float number.
    const int AVX_FLOAT_SIZE= 16; 
    // this should be the multiple of cache line size, but should not too large
    // the chunks of B matrix can fit inside the L1 cache.
    const int BLOCK_SIZE_X = 16;  
    // This value can be set larger, but also consider the size of L1.
    const int BLOCK_SIZE_Y = 64;
    size_t mat_size = N*N*sizeof(float);
    memset(out,0,mat_size);    

    #pragma omp parallel for schedule(dynamic,1)
    for(int i=0;i<N;i+=BLOCK_SIZE_Y){
        for(int j=0;j<N;j+=BLOCK_SIZE_X){
            for(int k=0;k<N;k+=AVX_FLOAT_SIZE){
                for(int m=0;m<BLOCK_SIZE_Y;m++){
                    __m512 o = _mm512_load_ps(&out[i+m][k]);;
                    for(int n=0;n<BLOCK_SIZE_X;n++)
                    {
                        __m512 c = _mm512_set1_ps(A[i+m][j+n]);
                        __m512 b = _mm512_load_ps(&B[j+n][k]);
                        o = _mm512_fmadd_ps(c,b,o);
                    }
                    _mm512_store_ps(&out[i+m][k],o);
                }
            }
        }
    }
}
#endif

void compare_mat(float A[N][N],float B[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            if(fabs(A[i][j]-B[i][j]) > 1e-5){
                printf("incorrect result!!!\n");
                return;
            }

}

float(* make_buffer())[N]
{
    const int buffer_size = N*N*sizeof(float);
    const int aliagn = 64;
    return (float(*)[N])std::aligned_alloc(aliagn, buffer_size);    
}

int main(int argc,char* argv[])
{
    const int N_1024 = N/1024;
    const int DEFAULT_LOOP = 100;
    const int LOOP_CNT = N_1024 <= 1? DEFAULT_LOOP : std::max(1, DEFAULT_LOOP/(N_1024*N_1024*N_1024));

    auto A = make_buffer();
    auto B = make_buffer();
    auto C1= make_buffer();
    auto C2= make_buffer();
    auto C3 = make_buffer();

    mat_init(A);
    mat_init(B);

    printf("%dx%d matrix multiplication benchmark.\n",N,N);

    auto start = get_current_time_ms();
    if(N_1024<=1){
        matmul_base(C1,A,B);
        printf("matmul_base = %0.2f ms\n",(float)(get_current_time_ms()-start));
    }

    start = get_current_time_ms();
    for(int i=0;i<LOOP_CNT;i++)
        matmul_cache_friendly(C2,A,B);
    printf("matmul_cache_friendly = %0.2f ms\n",(float)(get_current_time_ms()-start)/LOOP_CNT);

    if(N_1024<=1) compare_mat(C1,C2);

#if AVX512F_CHECK
    start = get_current_time_ms();
    for(int i=0;i<LOOP_CNT;i++)
        matmul_avx512(C3,A,B);
    printf("matmul_avx512 = %0.2f ms\n",(float)(get_current_time_ms()-start)/LOOP_CNT);
    compare_mat(C2,C3);
#endif

    std::free(A);
    std::free(B);
    std::free(C1);
    std::free(C2);
    std::free(C3);
    return 0;
}