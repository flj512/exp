#include <chrono>
#include "utils.h"

int64_t get_current_time_us()
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}