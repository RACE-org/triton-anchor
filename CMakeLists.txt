cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(triton_anchor CXX C)

# The embedded Triton core/toolchain is provided by spine-triton-frontend under
# ./triton. triton-anchor keeps only the Python orchestration layer at the
# repository root; C++ lowering is built by the embedded spine toolchain.
add_subdirectory(triton)
