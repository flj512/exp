find_package(OpenMP QUIET)

add_library(common utils.cc)
if(OpenMP_CXX_FOUND)
    target_link_libraries(common PRIVATE OpenMP::OpenMP_CXX)
endif()
