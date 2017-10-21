cmake_minimum_required (VERSION 3.1)
project (MutualInformationRegistration)

# Find Eigen3
find_package(Eigen3 REQUIRED)

set(CCX g++)

# C++ compiler options
set(MIR_CXX_Options
    -std=c++14
    -Wconversion
    -Wall
    -Werror
    -pedantic
)

# Source files
set(LibMIR_SourceFiles
    src/libmir/mat.cpp
    src/libmir/math.cpp
    src/libmir/registration.cpp
    src/libmir/sat.cpp
)
set(MIRTool_SourceFiles
    src/mirtool/main.cpp
)

# Include folders
set(LibMIR_IncludeDirs
    ${EIGEN3_INCLUDE_DIR}
)
set(MIRTool_IncludeDirs
    ${EIGEN3_INCLUDE_DIR}
)

# Target binaries
add_library(libmir SHARED ${LibMIR_SourceFiles})
target_compile_options(libmir PUBLIC ${MIR_CXX_Options})
target_include_directories(libmir SYSTEM PUBLIC ${LibMIR_IncludeDirs})
target_include_directories(libmir SYSTEM PUBLIC src/third_party)

add_executable(miregister ${MIRTool_SourceFiles})
target_compile_options(miregister PUBLIC ${MIR_CXX_Options})
target_include_directories(miregister SYSTEM PUBLIC ${MIRTool_IncludeDirs})
target_include_directories(miregister SYSTEM PUBLIC src/third_party)
target_link_libraries(miregister libmir m stdc++)

