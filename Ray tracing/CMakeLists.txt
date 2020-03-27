cmake_minimum_required(VERSION 3.15)
project(RayTracing)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
add_executable(RayTracing main.cpp)