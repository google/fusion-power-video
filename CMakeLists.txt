# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.6)

project(FusionPowerVideo CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(PkgConfig REQUIRED)
pkg_check_modules(Brotli REQUIRED IMPORTED_TARGET libbrotlienc libbrotlidec)

add_library(fusion_power_video STATIC fusion_power_video.h fusion_power_video.cc)
target_link_libraries(fusion_power_video PRIVATE pthread PkgConfig::Brotli)

set_target_properties(fusion_power_video PROPERTIES PUBLIC_HEADER "fusion_power_video.h")
INSTALL(TARGETS fusion_power_video
        ARCHIVE DESTINATION lib 
        PUBLIC_HEADER DESTINATION include
)

foreach (executable IN ITEMS benchmark encode decode)
  add_executable("${executable}" "${executable}.cc")
  target_link_libraries("${executable}" fusion_power_video)
endforeach ()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/columnar_batch)

if (BUILD_ARROW_ENCODER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/arrow)
endif()
