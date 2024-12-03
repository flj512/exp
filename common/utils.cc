#include<chrono>
#include"utils.h"

int64_t get_current_time_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}