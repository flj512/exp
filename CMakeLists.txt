cmake_minimum_required(VERSION 3.22)
project(exp)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")  

set (CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wno-unused-variable)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-expansion-to-defined)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-unknown-pragmas)
else()
    message(FATAL_ERROR "Compiler ${CMAKE_CXX_COMPILER_ID} not fully tested.")
endif()

add_subdirectory(common)

include_directories(./)
link_libraries(common)

add_subdirectory(mat)
add_subdirectory(metaprogram)