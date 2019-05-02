cmake_minimum_required(VERSION 3.12)
project(ANNExample LANGUAGES C)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

find_library(ANN NAMES ANN)
if( NOT ANN)
    message(FATAL_ERROR "ANN library not found")
endif()
include_directories(${ANN_INCLUDE_DIR})


### FLAGS 
add_definitions(-Wall -Wextra -pedantic -Wformat=2 -std=c99)

set (CMAKE_C_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak -g3 -Og -fprofile-arcs -ftest-coverage -DDEBUG=1")
set (CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak")
set (CMAKE_MODULE_LINKER_FLAGS_DEBUG "-fsanitize=address  -fsanitize=leak")

set (CMAKE_C_FLAGS_RELEASE "-Ofast")
set (CMAKE_EXE_LINKER_FLAGS_RELEASE "-Ofast")
set (CMAKE_MODULE_LINKER_FLAGS_RELEASE "-Ofast")


### RULES

FILE(GLOB_RECURSE SRCFILES src/*.c)
add_executable(Main ${SRCFILES})
target_link_libraries(Main ${ANN_LIBRARIES})

### TARGET
add_custom_target(run
    COMMAND ${CMAKE_BINARY_DIR}/Main
    COMMENT "Running Main !!!"
    DEPENDS Main
)

add_custom_target(gdb
    COMMAND gdb ${CMAKE_BINARY_DIR}/Main
    COMMENT "Running Main with gdb !!!"
    DEPENDS Main
)
