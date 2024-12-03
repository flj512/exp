cmake_minimum_required(VERSION 3.0)
project(exp)

add_compile_options(-Wall)

add_subdirectory(common)

include_directories(./)
link_libraries(common)

add_subdirectory(mat)