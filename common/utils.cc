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
#include <thread>
#include <cstdlib>
#include <string>

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
        printf("bind cpu failed\n");
        exit(1);
    }
#else
#pragma warning("bind_cpu is not implemented on this platform");
#endif
}
void limit_max_num_threads()
{
#ifdef _OPENMP
    int max_threads = std::thread::hardware_concurrency();
    int current_threads = max_threads;

    std::string key = "OMP_NUM_THREADS";
    char *env = std::getenv(key.c_str());
    if (env)
    {
        current_threads = std::atoi(env);
    }

    if (current_threads > max_threads / 2)
    {
        current_threads = max_threads / 2;
        omp_set_num_threads(current_threads);
    }

    printf("Using omp num threads = %d, max = %d\n", current_threads, max_threads);
#endif
}