cmake_minimum_required(VERSION 3.10)
project(rt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
find_package(Threads REQUIRED)
add_executable(rt main.cpp Bitmap.cpp)
target_link_libraries(rt ${CMAKE_THREAD_LIBS_INIT})