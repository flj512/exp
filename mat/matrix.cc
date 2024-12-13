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

#define N (1024) // only test NxN matrix, N is multiple of 16

float* make_buffer()
{
    const int buffer_size = N*N*sizeof(float);
    const int aliagn = 64;
    return (float*)std::aligned_alloc(aliagn, buffer_size);    
}

void print_matrix(const float* m,int n,int s, const char * tag=nullptr)
{
    printf("***%s***\n",tag);

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            printf("%0.1f\t\t",m[i*s+j]);
        printf("\n");
    }
}

void mat_init(float *m, int n) // nxn mat, stride = n
{
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            m[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

void mat_clear(float *m, int n, int s)
{
    for(int i=0;i<n;i++)
        memset(m+i*s,0,n*sizeof(float));
}

void matmul_base(float* out, const float* A,const float* B, int n, int s)
{
    mat_clear(out,n,s);

    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            for(int k=0;k<n;k++)
                out[i*s+j] += A[i*s+k]*B[k*s+j];
}
void matadd_base(float* out, const float* A,const float* B, int n, int s)
{
    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            out[i*s+j]=A[i*s+j]+B[i*s+j];
}

void matsub_base(float* out, const float* A,const float* B, int n, int s)
{
    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            out[i*s+j]=A[i*s+j]-B[i*s+j];
}

void matmul_cache_friendly(float* out, const float* A,const float* B, int n, int s)
{
    mat_clear(out,n,s);

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

    mat_clear(out,n,s);    

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

void matadd_avx512(float* out, const float* A,const float* B, int n, int s)
{
    const int AVX_FLOAT_SIZE= 16;

    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j+=AVX_FLOAT_SIZE){
            __m512 a = _mm512_load_ps(&A[i*s+j]);
            __m512 b = _mm512_load_ps(&B[i*s+j]);
            __m512 o = _mm512_add_ps(a,b);
            _mm512_store_ps(&out[i*s+j],o);
        }
}

void matsub_avx512(float* out, const float* A,const float* B, int n, int s)
{
    const int AVX_FLOAT_SIZE= 16;

    #pragma omp parallel for
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j+=AVX_FLOAT_SIZE){
            __m512 a = _mm512_load_ps(&A[i*s+j]);
            __m512 b = _mm512_load_ps(&B[i*s+j]);
            __m512 o = _mm512_sub_ps(a,b);
            _mm512_store_ps(&out[i*s+j],o);
        }
}
#endif

/*
    B0, B1
    B2, B3
*/
template<typename T>
inline T* mat_block(int index,T* A,int n,int s)
{
    return A + (index>>1)*(n>>1)*s+(index&1)*(n>>1);
}

void matmul_strassen_(float* out, const float* A,const float* B, int n, int s,float* buffer)
{
    const int MIN_SIZE = 512;
#if AVX512F_CHECK
    auto matadd = matadd_avx512;
    auto matsub = matsub_avx512;
    auto matmul = matmul_avx512;
#else
    auto matadd = matadd_base;
    auto matsub = matsub_base;
    auto matmul = matmul_cache_friendly;
#endif
    
    if(n<=MIN_SIZE){
        return matmul(out,A,B,n,s);
    }

    mat_clear(out,n,s);
    /*
        A = a b
            c d
        B = e f
            g h
        buffer = b0 b1
                 b2 b3
        out = M1 + M4 - M5 + M7,  M3 + M5
              M2 + M4          ,  M1 + M3 - M2 + M6
    */
    auto a = mat_block(0,A,n,s);
    auto b = mat_block(1,A,n,s);
    auto c = mat_block(2,A,n,s);
    auto d = mat_block(3,A,n,s);
    auto e = mat_block(0,B,n,s);
    auto f = mat_block(1,B,n,s);
    auto g = mat_block(2,B,n,s);
    auto h = mat_block(3,B,n,s);
    auto b0 = mat_block(0,buffer,n,s);
    auto b1 = mat_block(1,buffer,n,s);
    auto b2 = mat_block(2,buffer,n,s);
    auto b3 = mat_block(3,buffer,n,s);
    auto out0 = mat_block(0,out,n,s);
    auto out1 = mat_block(1,out,n,s);
    auto out2 = mat_block(2,out,n,s);
    auto out3 = mat_block(3,out,n,s);
    // M1 = (a+d)(e+h)
    matadd(b0,a,d,n/2,s); // a + d = b0
    matadd(b1,e,h,n/2,s); // e + h = b1
    matmul_strassen_(b2,b0,b1,n/2,s,b3); // M1 = b2 = b0*b1, b3 is the buffer in sub matrix multiplication.
    //add M1 to out0, out3
    matadd(out0,out0,b2,n/2,s);
    matadd(out3,out3,b2,n/2,s);


    // M2 = (c+d)e
    matadd(b0,c,d,n/2,s); // c + d = b0
    matmul_strassen_(b2,b0,e,n/2,s,b3); // M2 = b2 = b0*e, b3 is the buffer in sub matrix multiplication.
    //add M1 to out2, -M2 to out3
    matadd(out2,out2,b2,n/2,s);
    matsub(out3,out3,b2,n/2,s);

    // M3 = a(f-h)
    matsub(b0,f,h,n/2,s); // f - h = b0
    matmul_strassen_(b2,a,b0,n/2,s,b3); // M3 = b2 = b0*a, b3 is the buffer in sub matrix multiplication.
    //add M3 to out1, out3
    matadd(out1,out1,b2,n/2,s);
    matadd(out3,out3,b2,n/2,s);

    // M4 = d(g-e)
    matsub(b0,g,e,n/2,s); // g - e = b0
    matmul_strassen_(b2,d,b0,n/2,s,b3); // M4 = b2 = b0*d, b3 is the buffer in sub matrix multiplication.
    //add M4 to out0, out2
    matadd(out0,out0,b2,n/2,s);
    matadd(out2,out2,b2,n/2,s);

    // M5 = (a+b)h
    matadd(b0,a,b,n/2,s); // a + b = b0
    matmul_strassen_(b2,b0,h,n/2,s,b3); // M5 = b2 = b0*h, b3 is the buffer in sub matrix multiplication.
    //add -M5 to out0, M5 to out1
    matsub(out0,out0,b2,n/2,s);
    matadd(out1,out1,b2,n/2,s);

    // M6 = (c-a)(e+f)
    matsub(b0,c,a,n/2,s); // c - a = b0
    matadd(b1,e,f,n/2,s); // e + f = b1
    matmul_strassen_(b2,b0,b1,n/2,s,b3); // M6 = b2 = b0*b1, b3 is the buffer in sub matrix multiplication.
    //add M6 to out3
    matadd(out3,out3,b2,n/2,s);

    // M7 = (b - d)(g + h)
    matsub(b0,b,d,n/2,s); // b - d = b0
    matadd(b1,g,h,n/2,s); // g + h = b1
    matmul_strassen_(b2,b0,b1,n/2,s,b3); // M7 = b2 = b0*b1, b3 is the buffer in sub matrix multiplication.
    //add M7 to out0
    matadd(out0,out0,b2,n/2,s);
}

void matmul_strassen(float* out, const float* A,const float* B, int n, int s)
{
    auto buffer = make_buffer(); // it's ok, because n = N, s=N in this benchmark
    matmul_strassen_(out,A,B,n,s,buffer);
    std::free(buffer);
}

void compare_mat(const float* A,const float* B,int n,int s)
{
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++){
            auto a = A[i*s+j];
            auto b = B[i*s+j];
            float err = fabs(a) > 1 ? fabs(a-b)/fabs(a):fabs(a-b);
            if(err > 1e-4){
                printf("incorrect result,(%f != %f,error=%f) @ [%d,%d]!!!\n",a,b,err,i,j);
                return;
            }
        }
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
    auto BC1= make_buffer();
    auto BC2= make_buffer();
    auto BC= make_buffer();
    auto C= make_buffer();

    mat_init(A,N);
    mat_init(B,N);

    printf("%dx%d matrix multiplication benchmark.\n",N,N);

    if(N_1024<=1){
        BENCHMARK_FUNCTION(BC,A,B,matmul_base,N,N,nullptr,1)
    }

    BENCHMARK_FUNCTION(C,A,B,matmul_cache_friendly,N,N,N_1024<=1?BC:nullptr,LOOP_CNT)
    BENCHMARK_FUNCTION(BC1,A,B,matadd_base,N,N,nullptr,DEFAULT_LOOP)
    BENCHMARK_FUNCTION(BC2,A,B,matsub_base,N,N,nullptr,DEFAULT_LOOP)

#if AVX512F_CHECK
    BENCHMARK_FUNCTION(BC,A,B,matadd_avx512,N,N,BC1,DEFAULT_LOOP)
    BENCHMARK_FUNCTION(BC,A,B,matsub_avx512,N,N,BC2,DEFAULT_LOOP)
    BENCHMARK_FUNCTION(BC,A,B,matmul_avx512,N,N,C,LOOP_CNT)
#endif

    BENCHMARK_FUNCTION(BC,A,B,matmul_strassen,N,N,C,LOOP_CNT)

    std::free(A);
    std::free(B);
    std::free(BC);
    std::free(BC1);
    std::free(BC2);
    std::free(C);
    return 0;
}