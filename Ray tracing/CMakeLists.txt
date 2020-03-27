cmake_minimum_required(VERSION 3.10)
project(rt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
add_executable(rt main.cpp Bitmap.cpp)
