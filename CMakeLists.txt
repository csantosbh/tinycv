cmake_minimum_required (VERSION 3.1)
project (TinyCV)

set(CCX g++)

# C++ compiler options
set(TinyCV_CXX_Options
    -std=c++14
    -Wconversion
    -Wall
    -Werror
    -pedantic
)

# Sample tools
set(MIRTool_SourceFiles
    src/samples/mutual_information_registration.cpp
)

# Include folders
set(TinyCV_IncludeDirs
    src/
)

# Target binaries
add_executable(mirtool ${MIRTool_SourceFiles})
target_compile_options(mirtool PUBLIC ${TinyCV_CXX_Options})
target_include_directories(mirtool SYSTEM PUBLIC ${TinyCV_IncludeDirs})
target_link_libraries(mirtool m stdc++)

