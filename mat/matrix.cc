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

#define N 1024

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

void matmul_cacheline(float out[N][N], const float A[N][N],const float B[N][N])
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
    const int BLOCK_SIZE= 16;
    size_t mat_size = N*N*sizeof(float);
    memset(out,0,mat_size);    

    #pragma omp parallel for
    for(int i=0;i<N;i++)
    {
        for(int j=0;j<N;j+=BLOCK_SIZE){
            __m512 a = _mm512_load_ps(&A[i][j]);
            for(int k=0;k<N;k+=BLOCK_SIZE){
                __m512 o = _mm512_load_ps(&out[i][k]);;
                for(int l=0;l<BLOCK_SIZE;l++)
                {
                    __m512 c = _mm512_set1_ps(a[l]);
                    __m512 b = _mm512_load_ps(&B[j+l][k]);
                    o = _mm512_fmadd_ps(c,b,o);
                }
                _mm512_store_ps(&out[i][k],o);
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

alignas(64) float A[N][N],B[N][N];
alignas(64) float C1[N][N],C2[N][N],C3[N][N];

int main(int argc,char* argv[])
{
    const int LOOP_CNT = 50;
    mat_init(A);
    mat_init(B);


    auto start = get_current_time_ms();
    for(int i=0;i<LOOP_CNT;i++)
        matmul_base(C1,A,B);
    printf("matmul_base = %0.2f ms\n",(float)(get_current_time_ms()-start)/LOOP_CNT);

    start = get_current_time_ms();
    for(int i=0;i<LOOP_CNT;i++)
        matmul_cacheline(C2,A,B);
    printf("matmul_cacheline = %0.2f ms\n",(float)(get_current_time_ms()-start)/LOOP_CNT);

    compare_mat(C1,C2);

#if AVX512F_CHECK
    start = get_current_time_ms();
    for(int i=0;i<LOOP_CNT;i++)
        matmul_avx512(C3,A,B);
    printf("matmul_avx512 = %0.2f ms\n",(float)(get_current_time_ms()-start)/LOOP_CNT);
    compare_mat(C2,C3);
#endif

    return 0;
}