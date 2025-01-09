#include <omp.h>

#include <cstdio>
#include <thread>

#include "common/utils.h"

void test_static_sched() {
  printf("====== static test ======\n");
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 8; i++) {
    printf("[%d] thread id=%d\n", i, omp_get_thread_num());
  }
}
void test_dynamic_sched() {
  printf("====== dynamic test ======\n");
#pragma omp parallel for num_threads(2) schedule(dynamic, 1)
  for (int i = 0; i < 8; i++) {
    printf("[%d] thread id=%d\n", i, omp_get_thread_num());
  }
}
void test_barrier_sched() {
  int number_threads = 2;
  printf("====== barrier test ======\n");
#pragma omp parallel num_threads(number_threads)
  {
    int thread_id = omp_get_thread_num();
    for (int i = thread_id; i < 8; i += number_threads) {
      printf("[%d] thread id=%d\n", i, thread_id);
#pragma omp barrier
    }
  }
}

void mat_add(float *out, float *A, float *B, int n, int s)  // nxn mat
{
#pragma omp parallel for  // DON'T FORGET TO ADD 'for'
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) out[i * s + j] = A[i * s + j] + B[i * s + j];
}

void test_mat_add() {
  printf("====== mat add test ======\n");
  const int N = 4;
  const int S = N * N;
  float A[S];
  float B[S];

  for (int i = 0; i < S; i++) {
    A[i] = i + 1;
    B[i] = 1.0;
  }

  mat_add(A, A, B, N, N);
  for (int i = 0; i < S; i++) {
    printf("%0.1f ", A[i]);
  }
  printf("\n");
}
void test_bind_cpu(int n) {
  printf("====== bind cpu test, n = %d ======\n", n);
#pragma omp parallel for num_threads(2)
  for (int i = 0; i < n; i++) {
    bind_cpu();
    for (int j = 0; j < 200; j++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::hash<std::thread::id> hasher;
    printf("[%d] thread id=%lu, omp id = %d, cpu_id = %d\n", i,
           hasher(std::this_thread::get_id()), omp_get_thread_num(), get_cpu());
  }
}
int main() {
  test_static_sched();
  test_dynamic_sched();
  test_barrier_sched();
  test_mat_add();
  test_bind_cpu(8);
  test_bind_cpu(32);
  return 0;
}