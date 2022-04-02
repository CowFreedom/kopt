#See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#fn_2
#and https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/

cmake_minimum_required(VERSION 3.9...3.23)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project (GenerateRandomNumbers VERSION 1.0
				DESCRIPTION "Generates random numbers"
				LANGUAGES CXX CUDA) 

#SET(KFRAND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}../../../src/krand/device/cuda" CACHE INTERNAL "")

FIND_PATH(KFRAND_INCLUDE_DIR 
		NAMES kf_random.h
		PATHS ${CMAKE_CURRENT_SOURCE_DIR}../../../src/krand/device/cuda
		PATH_SUFFIXES "/device/cuda")

FIND_PATH(KFPCG_INCLUDE_DIR 
		NAMES pcg.h
		PATHS ${CMAKE_CURRENT_SOURCE_DIR}../../../src/device/cuda
		PATH_SUFFIXES "/device/cuda")

message(STATUS "KFRAND_INCLUDE_DIR => ${KFRAND_INCLUDE_DIR}")

add_executable (generate_random_numbers main.cu ${KFRAND_INCLUDE_DIR}/kf_random.h)
set_target_properties(generate_random_numbers PROPERTIES LINKER_LANGUAGE CUDA)

add_library(kfpcg STATIC ${KFPCG_INCLUDE_DIR}/pcg.cu ${KFPCG_INCLUDE_DIR}/pcg.h)
set_target_properties(kfpcg PROPERTIES LINKER_LANGUAGE CUDA)

target_include_directories(generate_random_numbers PUBLIC ${KFRAND_INCLUDE_DIR})
target_include_directories(kfpcg PUBLIC ${KFRAND_INCLUDE_DIR})


target_link_libraries(generate_random_numbers PUBLIC kfpcg)
#target_compile_options(kfpcg PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc -dlto -arch=sm_70>)
#target_compile_options(generate_random_numbers PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-dc -dlto -arch=sm_70>)
target_compile_features(generate_random_numbers PUBLIC cxx_std_17)
target_compile_features(kfpcg PUBLIC cxx_std_17)