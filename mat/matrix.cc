#include<cstring>
#include<cstdio>
#include<cmath>
#include<cassert>
#include<cstdlib>
#include<chrono>
#include"common/utils.h"

#define N 1024

void mat_init(float m[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            m[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void matmul_1(float out[N][N], const float A[N][N],const float B[N][N])
{
    size_t mat_size = N*N*sizeof(float);
    memset(out,0,mat_size);

    #pragma omp parallel for
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++)
                out[i][j] += A[i][k]*B[k][j];
}

void matmul_2(float out[N][N], const float A[N][N],const float B[N][N])
{
    size_t mat_size = N*N*sizeof(float);
    memset(out,0,mat_size);

    #pragma omp parallel for
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            for(int k=0;k<N;k++)
                out[i][k] += A[i][j]*B[j][k];
}


void compare_mat(float A[N][N],float B[N][N])
{
    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++)
            assert(fabs(A[i][j]-B[i][j]) < 1e-5);

}

float A[N][N],B[N][N];
float C1[N][N],C2[N][N];

int main(int argc,char* argv[])
{
    mat_init(A);
    mat_init(B);


    auto start = get_current_time_ms();
    matmul_2(C2,A,B);
    printf("matmul_2 = %ld ms\n",get_current_time_ms()-start);


    start = get_current_time_ms();
    matmul_1(C1,A,B);
    printf("matmul_1 = %ld ms\n",get_current_time_ms()-start);

    compare_mat(C1,C2);
    return 0;
}