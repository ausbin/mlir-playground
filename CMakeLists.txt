# I'm pretty sure this is terrible CMake but I don't care too much

project(mlir-playground LANGUAGES C CXX)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "-fuse-ld=gold")
endif()

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Don't bother showing warnings on macOS. Clang is too noisy
    add_compile_options(-Wall -Wextra -pedantic -fdiagnostics-color)
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs support core native)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
set(more_mlir_libs MLIRExecutionEngine)
set(all_libs ${llvm_libs} ${dialect_libs} ${more_mlir_libs})

add_executable(mlir-playground mlir-playground.cpp)
target_link_libraries(mlir-playground ${all_libs})
