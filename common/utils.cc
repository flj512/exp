#include <chrono>
#include <cstdio>
#include "utils.h"
#if defined(__linux__)
#include <sched.h>
#include <pthread.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

int64_t get_current_time_us()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int get_cpu()
{
#if defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}
void bind_cpu()
{
#if defined(__linux__) && defined(_OPENMP)
    int thread_num = omp_get_thread_num();
    int target_cpu = thread_num * 2;
    int cpu_id = get_cpu();
    if (cpu_id == target_cpu || cpu_id == -1)
    {
        return;
    }
#ifndef NDEBUG
    printf("bind thread %d to cpu %d, previous cpu = %d\n", thread_num, target_cpu, cpu_id);
#endif
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target_cpu, &cpuset);

    pthread_t current_thread = pthread_self();
    pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (get_cpu() != target_cpu)
    {
        printf("bind failed\n");
    }
#else
#pragma warning("bind_cpu is not implemented on this platform");
#endif
}