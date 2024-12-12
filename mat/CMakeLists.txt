find_package(OpenMP REQUIRED)

add_executable(matrix matrix.cc)

if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
endif()

# AVX512 support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
if(COMPILER_SUPPORTS_AVX512)
    message(STATUS "Compiler supports AVX512")
    target_compile_options(matrix PRIVATE -mavx512f)
endif()

# omp scheduler 
if(OpenMP_CXX_FOUND)
    add_executable(omp_sched omp_scheduler.cc)
    target_link_libraries(omp_sched PRIVATE OpenMP::OpenMP_CXX)
endif()