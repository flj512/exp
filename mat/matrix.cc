#include <cstring>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <memory>
#include "common/utils.h"
#define AVX512F_CHECK defined(__GNUC__) && defined(__AVX512F__)
#if AVX512F_CHECK
#include <immintrin.h>
#endif
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif

#define N (1024) // only test NxN matrix, N is multiple of 16

typedef std::shared_ptr<float> BufferPtr;
BufferPtr make_buffer(int n = N, int s = N)
{
    const int buffer_size = n * s * sizeof(float);
    const int aliagn = 64;
    return BufferPtr((float *)std::aligned_alloc(aliagn, buffer_size), [](float *p)
                     { std::free(p); });
}

void print_matrix(const float *m, int n, int s, const char *tag = nullptr)
{
    printf("***%s***\n", tag);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%0.1f\t\t", m[i * s + j]);
        printf("\n");
    }
}

#define ADD_OMP_THREADS 2
void mat_init(float *m, int n, bool random = true) // nxn mat, stride = n
{
    float d = 1.0f / n;
#pragma omp parallel for num_threads(ADD_OMP_THREADS)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            m[i * n + j] = random ? static_cast<float>(rand()) / static_cast<float>(RAND_MAX) : j * d;
}

void mat_clear(float *m, int n, int s)
{
    for (int i = 0; i < n; i++)
        memset(m + i * s, 0, n * sizeof(float));
}

// copy nxn matrix
void mat_copy(float *d, int sd, float *s, int ss, int n)
{
    for (int i = 0; i < n; i++)
        mempcpy(&d[i * sd], &s[i * ss], n * sizeof(float));
}

void avx_copy(void *d, void *s, size_t n)
{
    __m512i *dst = (__m512i *)d;
    const __m512i *src = (const __m512i *)s;

    for (; n > 0; n -= 512)
    {
        __m512i s0 = _mm512_load_si512(src + 0);
        __m512i s1 = _mm512_load_si512(src + 1);
        __m512i s2 = _mm512_load_si512(src + 2);
        __m512i s3 = _mm512_load_si512(src + 3);
        __m512i s4 = _mm512_load_si512(src + 4);
        __m512i s5 = _mm512_load_si512(src + 5);
        __m512i s6 = _mm512_load_si512(src + 6);
        __m512i s7 = _mm512_load_si512(src + 7);
        src += 8;

        _mm512_stream_si512(dst + 0, s0);
        _mm512_stream_si512(dst + 1, s1);
        _mm512_stream_si512(dst + 2, s2);
        _mm512_stream_si512(dst + 3, s3);
        _mm512_stream_si512(dst + 4, s4);
        _mm512_stream_si512(dst + 5, s5);
        _mm512_stream_si512(dst + 6, s6);
        _mm512_stream_si512(dst + 7, s7);
        dst += 8;
    }
}

void matmul_base(float *out, const float *A, const float *B, int n, int s)
{
    mat_clear(out, n, s);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                out[i * s + j] += A[i * s + k] * B[k * s + j];
}
void matadd_base(float *out, const float *A, const float *B, int n, int s)
{
#pragma omp parallel for num_threads(ADD_OMP_THREADS)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[i * s + j] = A[i * s + j] + B[i * s + j];
}

void matsub_base(float *out, const float *A, const float *B, int n, int s)
{
#pragma omp parallel for num_threads(ADD_OMP_THREADS)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            out[i * s + j] = A[i * s + j] - B[i * s + j];
}

#ifdef HAVE_OPENBLAS
void matadd_openblas(float *out, const float *A, const float *B, int n, int s)
{
    cblas_saxpy(n * n, 1.0f, A, 1, out, 1);
    cblas_saxpy(n * n, 1.0f, B, 1, out, 1);
}
void matmul_openblas(float *out, const float *A, const float *B, int n, int s)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    cblas_sgemm(CblasRowMajor, // Row-major layout
                CblasNoTrans,  // A is not transposed
                CblasNoTrans,  // B is not transposed
                n, n, n,       // Matrix dimensions: n x n
                alpha, A, n,   // Matrix A, with leading dimension N
                B, n,          // Matrix B, with leading dimension N
                beta, out, n); // Matrix out (result), with leading dimension N
}
#endif

void matmul_cache_friendly(float *out, const float *A, const float *B, int n, int s)
{
    mat_clear(out, n, s);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                out[i * s + k] += A[i * s + j] * B[j * s + k];
}

/*
    B0, B1
    B2, B3
*/
template <typename T>
inline T *mat_block(int index, T *A, int n, int s)
{
    return A + (index >> 1) * (n >> 1) * s + (index & 1) * (n >> 1);
}

/*
    divide to WxW block
    return (x,y) sub matrix
*/
template <typename T>
inline T *mat_block2(int x, int y, T *m, int n, int s, int B)
{
    int block_size = n / B;
    return m + x * block_size * s + y * block_size;
}

#if AVX512F_CHECK
void matadd_avx512(float *out, const float *A, const float *B, int n, int s)
{
    const int AVX_FLOAT_SIZE = 16;

#pragma omp parallel for num_threads(ADD_OMP_THREADS)
    for (int i = 0; i < n; i++)
    {
        bind_cpu();
        for (int j = 0; j < n; j += AVX_FLOAT_SIZE)
        {
            __m512 a = _mm512_load_ps(&A[i * s + j]);
            __m512 b = _mm512_load_ps(&B[i * s + j]);
            __m512 o = _mm512_add_ps(a, b);
            _mm512_store_ps(&out[i * s + j], o);
        }
    }
}

void matsub_avx512(float *out, const float *A, const float *B, int n, int s)
{
    const int AVX_FLOAT_SIZE = 16;

#pragma omp parallel for num_threads(ADD_OMP_THREADS)
    for (int i = 0; i < n; i++)
    {
        bind_cpu();
        for (int j = 0; j < n; j += AVX_FLOAT_SIZE)
        {
            __m512 a = _mm512_load_ps(&A[i * s + j]);
            __m512 b = _mm512_load_ps(&B[i * s + j]);
            __m512 o = _mm512_sub_ps(a, b);
            _mm512_store_ps(&out[i * s + j], o);
        }
    }
}

// out += AxB
void matfma_avx512_16x16(float *out, int so, const float *A, int sa, const float *B, int sb)
{
    for (int i = 0; i < 16; i++)
    {
        __m512 o = _mm512_load_ps(&out[i * so]);
        for (int j = 0; j < 16; j++)
        {
            __m512 a = _mm512_set1_ps(A[i * sa + j]);
            __m512 b = _mm512_load_ps(&B[j * sb]);
            o = _mm512_fmadd_ps(a, b, o);
        }
        _mm512_store_ps(&out[i * so], o);
    }
}
// out = AxB
void matmul_avx512_16x16(float *out, int so, const float *A, int sa, const float *B, int sb)
{
    for (int i = 0; i < 16; i++)
    {
        __m512 o = _mm512_set1_ps(0.0f);
        for (int j = 0; j < 16; j++)
        {
            __m512 a = _mm512_set1_ps(A[i * sa + j]);
            __m512 b = _mm512_load_ps(&B[j * sb]);
            o = _mm512_fmadd_ps(a, b, o);
        }
        _mm512_store_ps(&out[i * so], o);
    }
}

// out += AxB
void matfma_avx512_32x32(float *out, int so, const float *A, int sa, const float *B, int sb)
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
        {
            auto outij = mat_block(2 * i + j, out, 32, so);
            // A(i,0)*B(0,j)
            auto a0 = mat_block(2 * i, A, 32, sa);
            auto b0 = mat_block(j, B, 32, sb);
            matfma_avx512_16x16(outij, so, a0, sa, b0, sb);

            // A(i,1)*B(1,j)
            auto a1 = mat_block(2 * i + 1, A, 32, sa);
            auto b1 = mat_block(2 + j, B, 32, sb);
            matfma_avx512_16x16(outij, so, a1, sa, b1, sb);
        }
}

void matmul_avx512_block(float *out, const float *A, const float *B, int n, int s)
{
    int BLOCK_NUM = n / 16;

    mat_clear(out, n, s);

#pragma omp parallel for
    for (int i = 0; i < BLOCK_NUM; i++)
    {
        for (int j = 0; j < BLOCK_NUM; j++)
        {
            for (int k = 0; k < BLOCK_NUM; k++)
            {
                auto outik = mat_block2(i, k, out, n, s, BLOCK_NUM);
                auto Aij = mat_block2(i, j, A, n, s, BLOCK_NUM);
                auto Bjk = mat_block2(j, k, B, n, s, BLOCK_NUM);
                matfma_avx512_16x16(outik, s, Aij, s, Bjk, s);
            }
        }
    }
}

void matmul_avx512(float *out, const float *A, const float *B, int n, int s)
{
    /*
        the trunk shape of A matrix is (BLOCK_Y, BLOCK_X)
        the trunk shape of B matrix is (BLOCK_X, AVX_FLOAT_SIZE)
    */
    // 512 bit vector can contain 16 single float number.
    const int AVX_FLOAT_SIZE = 16;
    // this should be the multiple of cache line size, but should not too large
    // the chunks of B matrix can fit inside the L1 cache.
    const int BLOCK_SIZE_X = 16;
    // This value can be set larger, but also consider the size of L1.
    const int BLOCK_SIZE_Y = std::max(1, std::min(n / 32, 64));

    mat_clear(out, n, s);

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i += BLOCK_SIZE_Y)
    {
        for (int j = 0; j < n; j += BLOCK_SIZE_X)
        {
            for (int k = 0; k < n; k += AVX_FLOAT_SIZE)
            {
                for (int m = 0; m < BLOCK_SIZE_Y; m++)
                {
                    __m512 o = _mm512_load_ps(&out[(i + m) * s + k]);
                    for (int p = 0; p < BLOCK_SIZE_X; p++)
                    {
                        __m512 c = _mm512_set1_ps(A[(i + m) * s + j + p]);
                        __m512 b = _mm512_load_ps(&B[(j + p) * s + k]);
                        o = _mm512_fmadd_ps(c, b, o);
                    }
                    _mm512_store_ps(&out[(i + m) * s + k], o);
                }
            }
        }
    }
}
#endif

void matmul_strassen_(float *out, const float *A, const float *B, int n, int s, float *buffer)
{
    const int MIN_SIZE = 1024;
#if AVX512F_CHECK
    auto matadd = matadd_avx512;
    auto matsub = matsub_avx512;
    auto matmul = matmul_avx512;
#else
    auto matadd = matadd_base;
    auto matsub = matsub_base;
    auto matmul = matmul_cache_friendly;
#endif

    if (n <= MIN_SIZE)
    {
        return matmul(out, A, B, n, s);
    }
    mat_clear(out, n, s);
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
    auto a = mat_block(0, A, n, s);
    auto b = mat_block(1, A, n, s);
    auto c = mat_block(2, A, n, s);
    auto d = mat_block(3, A, n, s);
    auto e = mat_block(0, B, n, s);
    auto f = mat_block(1, B, n, s);
    auto g = mat_block(2, B, n, s);
    auto h = mat_block(3, B, n, s);
    auto b0 = mat_block(0, buffer, n, s);
    auto b1 = mat_block(1, buffer, n, s);
    auto b2 = mat_block(2, buffer, n, s);
    auto b3 = mat_block(3, buffer, n, s);
    auto out0 = mat_block(0, out, n, s);
    auto out1 = mat_block(1, out, n, s);
    auto out2 = mat_block(2, out, n, s);
    auto out3 = mat_block(3, out, n, s);
    // M1 = (a+d)(e+h)
    matadd(b0, a, d, n / 2, s);                 // a + d = b0
    matadd(b1, e, h, n / 2, s);                 // e + h = b1
    matmul_strassen_(b2, b0, b1, n / 2, s, b3); // M1 = b2 = b0*b1, b3 is the buffer in sub matrix multiplication.
    // add M1 to out0, out3
    matadd(out0, out0, b2, n / 2, s);
    matadd(out3, out3, b2, n / 2, s);

    // M2 = (c+d)e
    matadd(b0, c, d, n / 2, s);                // c + d = b0
    matmul_strassen_(b2, b0, e, n / 2, s, b3); // M2 = b2 = b0*e, b3 is the buffer in sub matrix multiplication.
    // add M1 to out2, -M2 to out3
    matadd(out2, out2, b2, n / 2, s);
    matsub(out3, out3, b2, n / 2, s);

    // M3 = a(f-h)
    matsub(b0, f, h, n / 2, s);                // f - h = b0
    matmul_strassen_(b2, a, b0, n / 2, s, b3); // M3 = b2 = b0*a, b3 is the buffer in sub matrix multiplication.
    // add M3 to out1, out3
    matadd(out1, out1, b2, n / 2, s);
    matadd(out3, out3, b2, n / 2, s);

    // M4 = d(g-e)
    matsub(b0, g, e, n / 2, s);                // g - e = b0
    matmul_strassen_(b2, d, b0, n / 2, s, b3); // M4 = b2 = b0*d, b3 is the buffer in sub matrix multiplication.
    // add M4 to out0, out2
    matadd(out0, out0, b2, n / 2, s);
    matadd(out2, out2, b2, n / 2, s);

    // M5 = (a+b)h
    matadd(b0, a, b, n / 2, s);                // a + b = b0
    matmul_strassen_(b2, b0, h, n / 2, s, b3); // M5 = b2 = b0*h, b3 is the buffer in sub matrix multiplication.
    // add -M5 to out0, M5 to out1
    matsub(out0, out0, b2, n / 2, s);
    matadd(out1, out1, b2, n / 2, s);

    // M6 = (c-a)(e+f)
    matsub(b0, c, a, n / 2, s);                 // c - a = b0
    matadd(b1, e, f, n / 2, s);                 // e + f = b1
    matmul_strassen_(b2, b0, b1, n / 2, s, b3); // M6 = b2 = b0*b1, b3 is the buffer in sub matrix multiplication.
    // add M6 to out3
    matadd(out3, out3, b2, n / 2, s);

    // M7 = (b - d)(g + h)
    matsub(b0, b, d, n / 2, s);                 // b - d = b0
    matadd(b1, g, h, n / 2, s);                 // g + h = b1
    matmul_strassen_(b2, b0, b1, n / 2, s, b3); // M7 = b2 = b0*b1, b3 is the buffer in sub matrix multiplication.
    // add M7 to out0
    matadd(out0, out0, b2, n / 2, s);
}

void matmul_strassen(float *out, const float *A, const float *B, int n, int s)
{
    auto buffer = make_buffer(n, s);
    matmul_strassen_(out, A, B, n, s, buffer.get());
}

void mat_compare(const float *A, const float *B, int n, int s)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            auto a = A[i * s + j];
            auto b = B[i * s + j];
            float err = fabs(a) > 1 ? fabs(a - b) / fabs(a) : fabs(a - b);
            if (err > 1e-4)
            { // the error of strassen algorithm may be greater than 1e-5
                printf("incorrect result,(%f != %f,error=%f) @ [%d,%d]!!!\n", a, b, err, i, j);
                return;
            }
        }
}

void benchmark_fun(void(func)(float *, const float *, const float *, int, int), int cnt, const char *tag, int loop)
{
    std::vector<BufferPtr> A, B, C;
    for (int i = 0; i < cnt; i++)
    {
        A.push_back(make_buffer());
        B.push_back(make_buffer());
        C.push_back(make_buffer());
        mat_init(A.back().get(), N, false);
        mat_init(B.back().get(), N, false);
        mat_init(C.back().get(), N, false);
    }

    auto start = get_current_time_us();
    for (int l = 0; l < loop; l++)
        for (int i = 0; i < cnt; i++)
            func(C[i].get(), A[i].get(), B[i].get(), N, N);
    printf("%s = %0.4f ms, %d times\n", tag, (get_current_time_us() - start) / 1000.0f / (cnt * loop), cnt * loop);
}

void benchmark_mem()
{
    std::vector<BufferPtr> A, B;
    const int nSize = 1024;
    const int nLoop = 500;
    const int nIter = 10;
    const int64_t matSize = sizeof(float) * nSize * nSize;
    const float nTotalGB = matSize * nLoop * nIter / (1024.0 * 1024 * 1024);

    for (int i = 0; i < nLoop; i++)
    {
        A.push_back(make_buffer(nSize, nSize));
        B.push_back(make_buffer(nSize, nSize));
        mat_init(A.back().get(), nSize, false);
        mat_init(B.back().get(), nSize, false);
    }

    auto start = get_current_time_us();
    for (int n = 0; n < nIter; n++)
        for (int i = 0; i < nLoop; i++)
            memcpy(B[i].get(), A[i].get(), matSize);
    printf("memcpy speed = %0.4f GB/s, total Copied %0.1f GB\n", nTotalGB * 1000000 / (get_current_time_us() - start), nTotalGB);

    start = get_current_time_us();
    for (int n = 0; n < nIter; n++)
        for (int i = 0; i < nLoop; i++)
            avx_copy(B[i].get(), A[i].get(), matSize);
    printf("avxcpy speed = %0.4f GB/s, total Copied %0.1f GB\n", nTotalGB * 1000000 / (get_current_time_us() - start), nTotalGB);
}

#define BENCHMARK_FUNCTION(func, cnt) benchmark_fun(func, (cnt), #func, 1)
#define BENCHMARK_FUNCTION_LOOP(func, cnt, loop) benchmark_fun(func, (cnt), #func, (loop))

#define TEST_INIT(s)                                \
    int SIZE = s;                                   \
    std::vector<BufferPtr> buffers;                 \
    auto alloctor = [&buffers, SIZE]() -> float * { \
        buffers.push_back(make_buffer(SIZE, SIZE)); \
        return buffers.back().get();                \
    };                                              \
    auto A = alloctor();                            \
    auto B = alloctor();                            \
    auto C1 = alloctor();                           \
    auto C2 = alloctor();                           \
    mat_init(A, SIZE);                              \
    mat_init(B, SIZE);                              \
    matmul_base(C1, A, B, SIZE, SIZE)

#define CHECK_FUN(fun)          \
    mat_clear(C2, SIZE, SIZE);  \
    printf("check " #fun "\n"); \
    fun(C2, A, B, SIZE, SIZE);  \
    mat_compare(C1, C2, SIZE, SIZE)

#if AVX512F_CHECK
void check_16()
{
    TEST_INIT(16);

    auto matfma_avx512_16x16_ = [](float *out, const float *A, const float *B, int n, int s)
    {
        matfma_avx512_16x16(out, s, A, s, B, s);
    };
    CHECK_FUN(matfma_avx512_16x16_);

    auto matmul_avx512_16x16_ = [](float *out, const float *A, const float *B, int n, int s)
    {
        matmul_avx512_16x16(out, s, A, s, B, s);
    };
    CHECK_FUN(matmul_avx512_16x16_);
}
void check_32()
{
    TEST_INIT(32);

    auto matfma_avx512_32x32_ = [](float *out, const float *A, const float *B, int n, int s)
    {
        matfma_avx512_32x32(out, s, A, s, B, s);
    };
    CHECK_FUN(matfma_avx512_32x32_);
}
#endif

void check_n(int n)
{
    printf("check %dx%d ...\n", n, n);

    TEST_INIT(n);

    CHECK_FUN(matmul_cache_friendly);

#if AVX512F_CHECK
    CHECK_FUN(matmul_avx512);
    CHECK_FUN(matmul_avx512_block);
#endif
#ifdef HAVE_OPENBLAS
    CHECK_FUN(matmul_openblas);
#endif
    CHECK_FUN(matmul_strassen);
}

void check_correct()
{
#if AVX512F_CHECK
    check_16();
    check_32();
#endif

    for (int n = 32; n <= 1024; n *= 2)
        check_n(n);
    ;
}
int main(int argc, char *argv[])
{
    check_correct();

    const float N_1024 = N / 1024.0;
    const int DEFAULT_LOOP = 100;
    const int MUL_LOOP_CNT = (int)(N_1024 <= 1 ? DEFAULT_LOOP / (N_1024 * N_1024) : std::max(1.0f, DEFAULT_LOOP / (N_1024 * N_1024 * N_1024)));
    const int ADD_LOOP_CNT = (int)std::max(1.0f, DEFAULT_LOOP * 4 / (N_1024 * N_1024));

    printf("\n------ %dx%d matrix benchmark ------\n", N, N);

    if (N_1024 <= 1)
    {
        BENCHMARK_FUNCTION(matmul_base, 1);
    }

    BENCHMARK_FUNCTION(matmul_cache_friendly, MUL_LOOP_CNT);
    BENCHMARK_FUNCTION(matsub_base, ADD_LOOP_CNT);
    BENCHMARK_FUNCTION(matadd_base, ADD_LOOP_CNT);
#ifdef HAVE_OPENBLAS
    BENCHMARK_FUNCTION(matadd_openblas, ADD_LOOP_CNT);
    BENCHMARK_FUNCTION(matmul_openblas, MUL_LOOP_CNT);
#endif

#if AVX512F_CHECK
    BENCHMARK_FUNCTION(matadd_avx512, ADD_LOOP_CNT);
    BENCHMARK_FUNCTION(matsub_avx512, ADD_LOOP_CNT);
    BENCHMARK_FUNCTION(matmul_avx512_block, MUL_LOOP_CNT);
    BENCHMARK_FUNCTION(matmul_avx512, MUL_LOOP_CNT);
#endif

    BENCHMARK_FUNCTION(matmul_strassen, MUL_LOOP_CNT);

    benchmark_mem();
    return 0;
}