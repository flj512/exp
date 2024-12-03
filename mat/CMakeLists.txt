find_package(OpenMP REQUIRED)

add_executable(matrix matrix.cc)

if(OpenMP_CXX_FOUND)
    target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
endif()