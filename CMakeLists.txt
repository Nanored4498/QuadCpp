project(quad)

cmake_minimum_required(VERSION 2.6)
cmake_policy(VERSION 2.6)

#--------#
# OPTION #
#--------#

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-long-long")
option(USE_OPENMP "OpenMP?"      OFF)
set(CMAKE_BUILD_TYPE Release)

#----------#
# Packages #
#----------#

if(USE_OPENMP)
    find_package(OpenMP)
endif(USE_OPENMP)
if(USE_OPENMP AND OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

#-------#
# Files #
#-------#

set(SRCS main.cpp)

#----------#
# INCLUDES #
#----------#

include_directories(${OPENGL_INCLUDE_DIR})
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#-------#
# Build #
#-------#

add_executable(${PROJECT_NAME} ${SRCS})
target_link_libraries(${PROJECT_NAME}
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARY}
)