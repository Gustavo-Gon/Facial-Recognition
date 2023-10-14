cmake_minimum_required(VERSION 3.25)
project(FaceRec)

set(CMAKE_CXX_STANDARD 17)  # Set the C++ standard to C++17

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set the path to dlib package
set(dlib_DIR "/opt/homebrew/Cellar/dlib/19.24_1/share/dlib")

# Find dlib package
find_package(dlib REQUIRED)

# Add the executable and specify the source files
add_executable(FaceRec main.cpp)

# Link OpenCV and dlib libraries to your project
target_link_libraries(FaceRec PRIVATE ${OpenCV_LIBS} dlib::dlib)
