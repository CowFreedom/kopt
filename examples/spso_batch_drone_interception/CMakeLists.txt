#See https://cliutils.gitlab.io/modern-cmake/chapters/basics.html#fn_2
#and https://developer.nvidia.com/blog/improving-gpu-app-performance-with-cuda-11-2-device-lto/

cmake_minimum_required(VERSION 3.9...3.23)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project (spso_batch_drone_interception VERSION 1.0
				DESCRIPTION "Finds the optimal angle for a shell shot from a tube to intercept a drone. Solved via particle swarm optimization"
				LANGUAGES CXX CUDA) 

FIND_PATH(KRAND_INCLUDE_DIR 
		NAMES krand.h
		PATHS ${CMAKE_CURRENT_SOURCE_DIR}../../../src/ext/krand/src/krand/
		PATH_SUFFIXES "/core/")

FIND_PATH(KOPT_INCLUDE_DIR 
		NAMES kopt.h
		PATHS ${CMAKE_CURRENT_SOURCE_DIR}../../../src/kopt/
		PATH_SUFFIXES "/core/")

message(STATUS "KRAND_INCLUDE_DIR => ${KRAND_INCLUDE_DIR}")

message(STATUS "KOPT_INCLUDE_DIR => ${KOPT_INCLUDE_DIR}")

add_executable(spso_batch_drone_interception ${CMAKE_CURRENT_SOURCE_DIR}/spso_batch_drone_interception.cu ) #${KOPT_INCLUDE_DIR}/kopt.h ${KRAND_INCLUDE_DIR}/krand.h

set_target_properties(spso_batch_drone_interception PROPERTIES LINKER_LANGUAGE CUDA)
target_compile_definitions(spso_batch_drone_interception PUBLIC USE_CUDA=1)

target_include_directories(spso_batch_drone_interception PUBLIC ${KOPT_INCLUDE_DIR}../../  ${KRAND_INCLUDE_DIR}../../)
target_compile_features(spso_batch_drone_interception PUBLIC cxx_std_17)