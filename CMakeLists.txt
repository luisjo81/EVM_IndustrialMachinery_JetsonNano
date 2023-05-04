# CMakeList.txt : CMake project for evm_cpp_linux, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("evm-cpp-linux")

find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} "include" )

file(GLOB SOURCES "src/*.cpp" "src/*.c")

# Add source to this project's executable.
add_executable ( evm-cpp-linux ${SOURCES} )

# TODO: Add tests and install targets if needed.
if(OpenMP_CXX_FOUND)
    target_link_libraries( evm-cpp-linux PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
endif()

install(TARGETS evm-cpp-linux DESTINATION bin)