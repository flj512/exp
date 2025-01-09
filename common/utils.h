#pragma once
#include <cstdint>

int64_t get_current_time_us();
int get_cpu();
void bind_cpu();
void limit_max_num_threads();