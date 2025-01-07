find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(matrix matrix.cc)

if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
endif()

# openblas support
pkg_check_modules(OPENBLAS REQUIRED openblas64)
if(OPENBLAS_FOUND)
    target_compile_definitions(matrix PRIVATE HAVE_OPENBLAS)
    target_include_directories(matrix PRIVATE ${OPENBLAS_INCLUDE_DIRS})
    target_link_libraries(matrix PRIVATE ${OPENBLAS_LIBRARIES})
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
    add_executable(omp_test omp_test.cc)
    target_link_libraries(omp_test PRIVATE OpenMP::OpenMP_CXX)
endif()