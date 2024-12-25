cmake_minimum_required(VERSION 3.0)
project(exp)

set (CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall)

add_subdirectory(common)

include_directories(./)
link_libraries(common)

add_subdirectory(mat)
add_subdirectory(metaprogram)