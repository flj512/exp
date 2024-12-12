#include<omp.h>
#include<cstdio>

void test_static_sched()
{
    printf("====== static test ======\n");
    #pragma omp parallel for num_threads(2)
    for(int i=0;i<8;i++)
    {
        printf("[%d] thread id=%d\n",i,omp_get_thread_num());
    }
}
void test_dynamic_sched()
{
    printf("====== dynamic test ======\n");
    #pragma omp parallel for num_threads(2) schedule(dynamic,1)
    for(int i=0;i<8;i++)
    {
        printf("[%d] thread id=%d\n",i,omp_get_thread_num());
    }
}
void test_barrier_sched()
{
    int number_threads = 2;
    printf("====== barrier test ======\n");
    #pragma omp parallel num_threads(number_threads)
    {
        int thread_id = omp_get_thread_num();
        for(int i=thread_id;i<8;i+=number_threads){
            printf("[%d] thread id=%d\n",i,thread_id);
            #pragma omp barrier
        }
    }
}
int main()
{
    test_static_sched();
    test_dynamic_sched();
    test_barrier_sched();
    return 0;
}